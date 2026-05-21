// RUN: triton-opt --analyze-deps %s | FileCheck %s

// Test: MemRefType outside main_loop should pass
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_memref_outside_mainloop_pass(%arg0: memref<?xi8>, %arg1: memref<?xi8>) attributes {global_kernel = "local"} {
    %c0_i32 = arith.constant {ssbuffer.block_id = 15 : i32} 0 : i32
    %c1_i32 = arith.constant {ssbuffer.block_id = 15 : i32} 1 : i32
    %c10_i32 = arith.constant {ssbuffer.block_id = 15 : i32} 10 : i32
    %alloc = memref.alloc() {ssbuffer.block_id = 16 : i32} : memref<64xf32>
    scope.scope : () -> () {
      %0 = tensor.empty() {ssbuffer.block_id = 15 : i32} : tensor<64xf32>
      %1 = linalg.fill {ssbuffer.block_id = 15 : i32} ins(%c0_i32 : i32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
      %2 = scf.for %i = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%arg2 = %1) -> (tensor<64xf32>) : i32 {
        %3 = arith.constant {ssbuffer.block_id = 15 : i32} 1.0 : f32
        %4 = tensor.empty() {ssbuffer.block_id = 15 : i32} : tensor<64xf32>
        %5 = linalg.fill {ssbuffer.block_id = 15 : i32} ins(%3 : f32) outs(%4 : tensor<64xf32>) -> tensor<64xf32>
        scf.yield %5 : tensor<64xf32>
      } {ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}

// CHECK: module attributes
// CHECK: func.func @test_memref_outside_mainloop_pass
