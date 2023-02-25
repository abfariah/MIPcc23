import pyscipopt as scip

def compute_dual_gap(model: scip.Model) -> float:
    dual_gap = model.getGap()
    pb = model.getPrimalbound()
    db = model.getDualbound()
    if dual_gap > 1e10:  # If dual gap is infinite
        return 1
    if pb * db < 0:
        return 1
    else:
        true_gap = abs(pb - db) / max(abs(pb), abs(db))
        return true_gap