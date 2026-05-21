/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

static constexpr const char *DEBUG_TYPE = "analyze-deps";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) \
LLVM_DEBUG({ \
  DBGS(); \
  llvm::dbgs() << __VA_ARGS__; \
  llvm::dbgs() << "\n"; \
})

using namespace llvm;
using namespace mlir;
using namespace triton;
using namespace scope;
using namespace hivm;

namespace {

void collectNestedOps(Block *block, SmallVector<Operation *> &ops)
{
    for (auto &op : *block) {
        ops.push_back(&op);
        for (auto &region : op.getRegions())
            for (auto &innerBlock : region)
                collectNestedOps(&innerBlock, ops);
    }
}

// Collect operations with ssbuffer.main_loop attribute recursively in a region
static void collectMainLoopOpsRecursively(Region &region, SmallVector<Operation *> &mainLoopOps)
{
    for (Block &block : region) {
        for (Operation &op : block) {
            if (op.hasAttr("ssbuffer.main_loop")) {
                mainLoopOps.push_back(&op);
            }
            for (auto &nestedRegion : op.getRegions())
                collectMainLoopOpsRecursively(nestedRegion, mainLoopOps);
        }
    }
}

// Check if mainLoopOp's region contains MemRefType dependency values
// whose definingOp's parent is within mainLoopOp's region
static bool hasMemRefDepsInRegion(Operation *mainLoopOp)
{
    for (Region &region : mainLoopOp->getRegions()) {
        for (Block &block : region) {
            SmallVector<Operation *> allOps;
            collectNestedOps(&block, allOps);

            for (Operation *op : allOps) {
                for (Value operand : op->getOperands()) {
                    Operation *defOp = operand.getDefiningOp();
                    if (!defOp)
                        continue;
                    // Check if definingOp is within mainLoopOp's region
                    if (!region.isAncestor(defOp->getParentRegion()))
                        continue;
                    if (isa<MemRefType>(operand.getType())) {
                        LDBG("[ERROR]: Found MemRefType dependency in main_loop!\n");
                        return true;
                    }
                }
                for (Value result : op->getResults()) {
                    Operation *defOp = result.getDefiningOp();
                    if (!defOp)
                        continue;
                    if (!region.isAncestor(defOp->getParentRegion()))
                        continue;
                    if (isa<MemRefType>(result.getType())) {
                        LDBG("[ERROR]: Found MemRefType result in main_loop!\n");
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

} // namespace

bool checkMemRefDepsInMainLoop(ModuleOp module)
{
    bool hasMemRef = false;

    module.walk([&](ScopeOp scope) -> WalkResult {
        // Check if scope has VECTOR core type
        auto coreTypeAttr = scope->getAttrOfType<TCoreTypeAttr>(TCoreTypeAttr::name);
        if (!coreTypeAttr)
            return WalkResult::advance();

        if (coreTypeAttr.getTcoretype() != TCoreType::VECTOR) {
            return WalkResult::advance();
        }

        // Within VECTOR scope, find all ops with ssbuffer.main_loop attribute
        SmallVector<Operation *> mainLoopOps;
        collectMainLoopOpsRecursively(scope.getBodyRegion(), mainLoopOps);

        for (auto *mainLoopOp : mainLoopOps) {
            if (hasMemRefDepsInRegion(mainLoopOp)) {
                hasMemRef = true;
                return WalkResult::interrupt();
            }
        }

        return WalkResult::advance();
    });

    return hasMemRef;
}

void AnalyzeDepsPass::runOnOperation()
{
    ModuleOp module = getOperation();

    LDBG("Enter AnalyzeDeps pass");

    if (checkMemRefDepsInMainLoop(module)) {
        LDBG("AnalyzeDeps found MemRefType deps, signalFailure");
        signalPassFailure();
        return;
    }

    LDBG("AnalyzeDeps pass completed successfully");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeDepsPass()
{
    return std::make_unique<AnalyzeDepsPass>();
}

} // namespace triton
} // namespace mlir
