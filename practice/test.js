import * as onnx from 'onnxjs';
import * as tf from '@tensorflow/tfjs';

// // // Load the ONNX model
// const run=async ()=>{
// const session = new onnx.InferenceSession();
// await session.loadModel('test_model.onnx');

// // Create input tensor
// const inputTensor = new onnx.Tensor(new Float32Array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
//     0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]), 'float32', [1, 28]);

// // Run inference
// const output = await session.run([inputTensor]);
// console.log(output);
// }

// run()

// for(i=0;i<5;i++){
//     out=[]
//     ix=0
//     while(True){
//       xin=tf.oneHot()
//     }
//   }
console.log(tf.OneHot(tf.tensor([[0]])),28)

