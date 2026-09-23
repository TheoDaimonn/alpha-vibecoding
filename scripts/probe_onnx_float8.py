import onnx, onnxruntime as ort, numpy as np, ml_dtypes, json
from onnx import helper as h, numpy_helper as nh, TensorProto as T
results={}
for name,dtype,tp in [('fp8_e4m3',ml_dtypes.float8_e4m3fn,T.FLOAT8E4M3FN),('bf8_e5m2',ml_dtypes.float8_e5m2,T.FLOAT8E5M2)]:
 w=nh.from_array(np.ones((2,2),dtype=dtype),'w')
 for mode in ['native_matmul','cast_to_fp32']:
  nodes=([h.make_node('Cast',['w'],['wf'],to=T.FLOAT),h.make_node('MatMul',['x','wf'],['y'])] if mode=='cast_to_fp32' else [h.make_node('Cast',['x'],['xf'],to=tp),h.make_node('MatMul',['xf','w'],['yf']),h.make_node('Cast',['yf'],['y'],to=T.FLOAT)])
  model=h.make_model(h.make_graph(nodes,'probe',[h.make_tensor_value_info('x',T.FLOAT,[2,2])],[h.make_tensor_value_info('y',T.FLOAT,[2,2])],[w]),opset_imports=[h.make_opsetid('',19)],ir_version=9)
  try:
   sess=ort.InferenceSession(model.SerializeToString(),providers=['CPUExecutionProvider']); sess.run(None,{'x':np.ones((2,2),np.float32)})
   results[name+'_'+mode]='OK'
  except Exception as exc: results[name+'_'+mode]=str(exc)[:600]
print(json.dumps(results,indent=2))
