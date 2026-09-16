"""A minimal stand-in for an uproot TTree.

Discovery only needs each branch's name, ``typename`` and sub-leaf names, so a stub lets
the naming rules be tested against ntuple configurations no file on this machine has --
multiple track collections (``de``/``ue``/``dm``/``um``), configurable time-cluster and
line-seed names, extra MC step collections, and deliberate schema mismatches.
"""


class StubBranch:
    def __init__(self, typename, leaves=()):
        self.typename = typename
        self._leaves = list(leaves)

    def keys(self):
        return list(self._leaves)


class StubTree:
    def __init__(self, branches):
        # branches: {name: typename} or {name: (typename, [leaves])}
        self._branches = {}
        for name, value in branches.items():
            typename, leaves = value if isinstance(value, tuple) else (value, ())
            self._branches[name] = StubBranch(typename, leaves)

    def items(self, recursive=False):
        return list(self._branches.items())

    def keys(self, recursive=False):
        return list(self._branches)

    def __getitem__(self, name):
        return self._branches[name]


def vec(struct):
    return f"vector<mu2e::{struct}>"


def vecvec(struct):
    return f"std::vector<std::vector<mu2e::{struct}>>"


def obj(struct):
    return f"mu2e::{struct}"


def track_branches(tag, *, mc=True, hits=True, quals=("",), pids=1):
    """The branch set EventNtupleMaker writes for one track-fit config."""
    out = {tag: (vec("TrkInfo"), [f"{tag}.pdg", f"{tag}.nactive"]),
           f"{tag}segs": vecvec("TrkSegInfo"),
           f"{tag}segpars_lh": vecvec("LoopHelixInfo"),
           f"{tag}calohit": (vec("TrkCaloHitInfo"), [f"{tag}calohit.did"]),
           f"tcnt.n{tag}": "int32_t"}
    if hits:
        out[f"{tag}hits"] = vecvec("TrkStrawHitInfo")
        out[f"{tag}mats"] = vecvec("TrkStrawMatInfo")
    if mc:
        out[f"{tag}mc"] = (vec("TrkInfoMC"), [f"{tag}mc.nhits"])
        out[f"{tag}mcsim"] = vecvec("SimInfo")
        out[f"{tag}mcvd"] = vecvec("MCStepInfo")
        out[f"{tag}segsmc"] = vecvec("SurfaceStepInfo")
        if hits:
            out[f"{tag}hitsmc"] = vecvec("TrkStrawHitInfoMC")
    for leaf in quals:
        out[f"{tag}qual{leaf}"] = (vec("MVAResultInfo"), [f"{tag}qual{leaf}.result"])
    for i in range(pids):
        suffix = "" if i == 0 else str(i + 1)
        out[f"{tag}pid{suffix}"] = (vec("MVAResultInfo"), [f"{tag}pid{suffix}.result"])
    return out
