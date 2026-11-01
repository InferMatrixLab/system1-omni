# Sample traces

Recorded on 2026-10-25 with [`record_s1a.py`](../record_s1a.py) from
[System1-Agents](https://github.com/ThinkFlowLab/system1-agents) at `9b20b98`, against the
native Open-Jev-9B worker on one RTX 6000 Ada:

- `blackjack.trace.jsonl`: the first 20 hands of a 100-hand
  `s1a run blackjack --model jev --rethink off`.
- `ticket_router.trace.jsonl`: one run of `s1a run ticket_router --model jev --rethink off`
  over System1-Agents' 30 synthetic tickets.

The request records are as the recorder wrote them; that day's recorder marked every
finished episode `done`. The headers' `source`, `server` and `content` were filled in
afterwards, and the blackjack trace was cut to 20 episodes.

The prompt text and tickets are System1-Agents' work under the Apache License 2.0
([LICENSE](https://github.com/ThinkFlowLab/system1-agents/blob/9b20b989d919bc69227f81bc0ec571ca56925586/LICENSE),
[NOTICE](https://github.com/ThinkFlowLab/system1-agents/blob/9b20b989d919bc69227f81bc0ec571ca56925586/NOTICE));
the blackjack states come from RLCard.
