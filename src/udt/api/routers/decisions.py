"""Decision-log + approval routes (dev doc §11.2), module M10b.

```
GET  /decisions?episode_id=           decision log
GET  /approvals/pending               queue for the console
POST /approvals/{decision_id}         {approve|reject|modify, modified_action?}
```

See `api/schemas.py`'s `ApprovalOverrideRequest` docstring for the
disclosed scope of `POST /approvals/{decision_id}` in this slice: it
annotates a decision's *recorded* approval response for audit purposes,
it does not rewind the simulation or re-execute a different action (a
live, blocking pending-approval queue needs async coordination this
slice doesn't build — see that docstring for exactly what's deferred
and why).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from udt.api.deps import get_episode_manager
from udt.api.schemas import ApprovalOverrideRequest
from udt.logging.decision_log import DecisionRecord

router = APIRouter()


@router.get("/decisions", response_model=list[DecisionRecord])
def list_decisions(episode_id: str, req: Request) -> list[DecisionRecord]:
    session = get_episode_manager(req).get(episode_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"episode {episode_id!r} not found")
    return session.decisions


@router.get("/approvals/pending", response_model=list[DecisionRecord])
def list_pending_approvals(req: Request) -> list[DecisionRecord]:
    """**Disclosed:** an audit view — every decision, across every
    live episode, whose `approval.required` was `True` — not a live
    queue of decisions still awaiting a response (every decision this
    session's `EpisodeSession` produces has already been resolved
    headlessly by the time it's logged, see module docstring)."""
    pending: list[DecisionRecord] = []
    for session in get_episode_manager(req).all_sessions():
        pending.extend(d for d in session.decisions if d.approval.required)
    return pending


def _find_decision(req: Request, decision_id: str) -> DecisionRecord:
    for session in get_episode_manager(req).all_sessions():
        for record in session.decisions:
            if record.decision_id == decision_id:
                return record
    raise HTTPException(status_code=404, detail=f"decision {decision_id!r} not found")


@router.post("/approvals/{decision_id}", response_model=DecisionRecord)
def override_approval(
    decision_id: str, request: ApprovalOverrideRequest, req: Request
) -> DecisionRecord:
    record = _find_decision(req, decision_id)
    record.approval.response = request.response
    return record
