# Vendored pySLAM Sim3 solver

`Sim3Solver.{h,cpp}` is derived from pySLAM's ORB-SLAM3-style solver at
<https://github.com/luigifreda/pyslam/tree/master/cpp/solvers> (source blob
`8887103a88b1327c47fd49546fa9fbfb77b22572`). `Random.{h,cpp}` is the
associated random-number helper.

The vendored code retains its upstream notices and is licensed under GNU
GPLv3-or-later. It is intentionally isolated from the BSD-licensed native
Kimera-Multi-LCD sources pending a repository-wide licensing decision.

Local changes are limited to formatting, deterministic initialization, and
finite/degeneracy guards needed for safe library integration.
