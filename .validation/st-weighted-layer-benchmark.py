import json
import statistics
import sys
import time

import torch

from sentence_transformers.sentence_transformer.modules import WeightedLayerPooling


torch.set_num_threads(1)
revision = sys.argv[1]
reference = torch.load(sys.argv[2] + '.pt', weights_only=False) if len(sys.argv) > 2 else None
results = []
for num_layers, layer_start in [(13, 4), (25, 21)]:
    torch.manual_seed(29)
    layers = [torch.randn(8, 256, 768, requires_grad=True) for _ in range(num_layers)]
    pooling = WeightedLayerPooling(768, num_hidden_layers=num_layers - 1, layer_start=layer_start)
    storages = {}

    def pack(tensor):
        storage = tensor.untyped_storage()
        storages[storage.data_ptr()] = storage.nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = pooling({'all_layer_embeddings': layers})['token_embeddings']
    output.square().mean().backward()
    values = {'output': output.detach().clone(), 'weight_gradient': pooling.layer_weights.grad.clone(),
              'selected_gradients': [value.grad.clone() for value in layers[layer_start:]]}
    for layer in layers[:layer_start]:
        assert layer.grad is None or torch.count_nonzero(layer.grad) == 0
    if reference is not None:
        expected = reference[len(results)]['values']
        torch.testing.assert_close(values, expected, rtol=0, atol=0)
    times = []
    for repeat in range(9):
        for layer in layers:
            layer.grad = None
        pooling.zero_grad(set_to_none=True)
        start = time.perf_counter()
        pooling({'all_layer_embeddings': layers})['token_embeddings'].square().mean().backward()
        if repeat >= 2:
            times.append(time.perf_counter() - start)
    result = {'revision': revision, 'layers': num_layers, 'layer_start': layer_start,
              'median_seconds': statistics.median(times),
              'unique_saved_tensor_storage_bytes': sum(storages.values())}
    print(json.dumps(result), flush=True)
    result['values'] = values
    results.append(result)
torch.save(results, revision + '.pt')
if reference is not None:
    print(json.dumps([{'layers': old['layers'], 'speed_ratio': old['median_seconds'] / new['median_seconds'],
                       'saved_storage_ratio': old['unique_saved_tensor_storage_bytes'] / new['unique_saved_tensor_storage_bytes']}
                      for old, new in zip(reference, results)]), flush=True)
