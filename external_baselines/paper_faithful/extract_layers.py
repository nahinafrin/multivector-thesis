"""Write a reduced GGUF: all metadata (tokenizer etc.) + token_embd + blocks < N, block_count=N."""
import sys, gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType
src, dst, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
r = GGUFReader(src)
arch = r.fields["general.architecture"].contents()
w = GGUFWriter(dst, arch=arch)
for key, f in r.fields.items():
    if key.startswith("GGUF.") or key == "general.architecture":
        continue
    val = f.contents()
    if key == f"{arch}.block_count":
        val = N
    vtype = f.types[0]
    sub = f.types[-1] if vtype == GGUFValueType.ARRAY else None
    w.add_key_value(key, val, vtype, sub_type=sub)
keep = []
for t in r.tensors:
    n = t.name
    if n.startswith("blk."):
        if int(n.split(".")[1]) >= N: continue
    elif n not in ("token_embd.weight", "rope_freqs.weight", "output_norm.weight"):
        continue      # drop output_norm / output (not needed for hidden_states[N])
    keep.append(t)
for t in keep:
    w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("kept", [t.name for t in keep][:12], len(keep))
