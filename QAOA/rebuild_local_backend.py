from pathlib import Path
from torch.utils.cpp_extension import load

repo = Path('/home/quantum/ys_lee/Pauli-Propagation---GPU-acceleration')
src = repo / 'src_tensor' / 'pps_tensor_backend.cpp'
tmp = repo / 'src_tensor' / 'pps_tensor_backend_local_build.cpp'
build_dir = repo / '.build' / 'pps_backend_local_rebuild'
build_dir.mkdir(parents=True, exist_ok=True)

text = src.read_text(encoding='utf-8')
prelude = '#include <time.h>\nextern "C" int timespec_get(struct timespec*, int);\n'
tmp.write_text(prelude + text, encoding='utf-8')

mod = load(
    name='_pps_tensor_backend_local',
    sources=[str(tmp)],
    build_directory=str(build_dir),
    verbose=True,
    extra_cflags=['-O3', '-std=gnu++17', '-D_GLIBCXX_HAVE_TIMESPEC_GET=1', '-D_ISOC11_SOURCE'],
)
print('built module:', mod.__file__)
