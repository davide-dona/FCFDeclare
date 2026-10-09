# FCFDeclare

FCFDeclare (Fast Control-Flow Declare) mines Declare models from event logs and checks traces against them. It works on the control flow alone: a trace is a sequence of activity names, and every template is evaluated for a whole batch of traces at once with numpy.

## Installation

FCFDeclare needs Python 3.11 or newer and numpy 2.0 or newer.

```bash
uv add git+https://github.com/davide-dona/FCFDeclare
```

## Usage

```python
import fcfdeclare as fcf

log = [
    ['Register', 'Check', 'Approve', 'Close'],
    ['Register', 'Check', 'Reject', 'Close'],
    ['Register', 'Approve', 'Close'],
]

model = fcf.mine(
    log,
    activity_support=0.1,
    min_support=0.99,
    max_cardinality=2,
    vacuity=True,
)
model.save('model.json')
model = fcf.DeclareModel.load('model.json')

checker = fcf.Checker(model)
result = checker.check([['Register', 'Close'], ['Register', 'Approve', 'Close']])
result.satisfied  # how many constraints each trace satisfies, shape [B]
result.share  # the fraction of the constraints each trace satisfies, shape [B]
result.full  # 1.0 for a trace that satisfies every constraint, else 0.0, shape [B]

matrix = checker.holds(log)  # bool array [B, K], one column per constraint of the model
```

A `str` is itself a sequence of one-character strings, so `'abc'` is read as the trace of activities `a`, `b`, `c`.

Each call to `check` or `holds` costs a few numpy operations per template whatever the batch size, so pass all traces in one call rather than one trace per call. Duplicate traces in a batch are evaluated once.

## Templates

A constraint applies a template to one activity $a$ or to an ordered pair of distinct activities $(a, b)$. The **activation** of a binary template is the activity whose occurrence makes the constraint relevant to a trace.

| Template | A trace satisfies it when | Activation |
| --- | --- | --- |
| Existence$(a, n)$ | $a$ occurs at least $n$ times | |
| Absence$(a, n)$ | $a$ occurs fewer than $n$ times | |
| Exactly$(a, n)$ | $a$ occurs exactly $n$ times | |
| Init$(a)$ | the trace starts with $a$ | |
| End$(a)$ | the trace ends with $a$ | |
| Choice$(a, b)$ | $a$ or $b$ occurs | |
| Exclusive Choice$(a, b)$ | $a$ or $b$ occurs, but not both | |
| Responded Existence$(a, b)$ | $a$ occurs, and so does $b$ | $a$ |
| Not Responded Existence$(a, b)$ | $a$ occurs, and $b$ does not | $a$ |
| Response$(a, b)$ | $a$ occurs, and a $b$ follows every $a$ | $a$ |
| Precedence$(a, b)$ | $b$ occurs, and an $a$ precedes every $b$ | $b$ |
| Not Response$(a, b)$ | $a$ occurs, and no $b$ follows any $a$ | $a$ |
| Not Precedence$(a, b)$ | $b$ occurs, and no $a$ precedes any $b$ | $b$ |
| Chain Response$(a, b)$ | $a$ occurs, and a $b$ immediately follows every $a$ | $a$ |
| Chain Precedence$(a, b)$ | $b$ occurs, and an $a$ immediately precedes every $b$ | $b$ |
| Not Chain Response$(a, b)$ | $a$ occurs, and no $b$ immediately follows any $a$ | $a$ |
| Not Chain Precedence$(a, b)$ | $b$ occurs, and no $a$ immediately precedes any $b$ | $b$ |
| Alternate Response$(a, b)$ | $a$ occurs, and a $b$ follows every $a$ before the next $a$ | $a$ |
| Alternate Precedence$(a, b)$ | $b$ occurs, and an $a$ precedes every $b$ since the previous $b$ | $b$ |

Only Existence, Absence, and Exactly take a count $n$; the other templates have $n = 1$. An activity that a model does not name matches no constraint, but it still counts as an event: it can be the first or last event of a trace, and it separates two events for the Chain templates.

### Vacuity

Every template with an activation requires its activation to occur, as the table shows. With **vacuity** on, a trace in which the activation does not occur satisfies the constraint anyway, which is the usual reading of Declare: Response$(a, b)$ then says nothing about a trace without $a$. With vacuity off, such a trace violates the constraint. A model records the vacuity it was mined with, and its `Checker` applies the same one.

## Mining

`mine` builds the candidates in two steps and keeps those that enough traces satisfy. The **support** of a candidate is the fraction of the traces of the log that satisfy it.

1. An activity is **frequent** when it occurs in at least `activity_support` of the traces. The candidates are every unary template on every frequent activity, and every binary template on every ordered pair of distinct frequent activities. Existence, Absence, and Exactly are instantiated for $n = 1, \dots,$ `max_cardinality`.
2. A candidate is kept when its support is at least `min_support`. Both thresholds are inclusive.

`vacuity` decides whether vacuous satisfaction counts toward the support, as described above.

The constraints of a model are ordered by template in the order of the table, then by $n$, then by activity names. This order fixes the columns of `Checker.holds` and the saved file, so equal models save to identical bytes.

Absence$(a, 1)$ can never be mined when `min_support` exceeds `1 - activity_support`: a frequent $a$ occurs in at least `activity_support` of the traces, so Absence$(a, 1)$ holds in at most the rest of them.

### Relation to other Declare miners

The Apriori-based Declare miner of [Maggi, Bose, and van der Aalst (CAiSE 2012)](https://doi.org/10.1007/978-3-642-31095-9_18), which [Declare4Py](https://github.com/ivanDonadello/Declare4Py) implements, takes as binary candidates only the pairs of activities that occur together in enough traces. FCFDeclare takes every ordered pair of individually frequent activities, with no co-occurrence requirement. Templates that hold precisely when two activities do not co-occur, such as Not Responded Existence and Exclusive Choice, therefore become minable. The candidate set is larger, and the miner still evaluates all candidates of a template in one vectorized pass over the distinct traces of the log.

## Saved models

`DeclareModel.save` writes JSON:

```json
{
    "settings": {
        "activity_support": 0.1,
        "min_support": 0.99,
        "max_cardinality": 2,
        "vacuity": true
    },
    "constraints": [
        {"template": "Response", "activities": ["Register", "Close"], "n": 1, "support": 1.0}
    ]
}
```

`DeclareModel.load` accepts only this format and raises `ValueError` on a missing or extra key, an invalid setting or constraint, a repeated constraint, or a support outside $[0, 1]$.

## Development

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
```

## License

MIT
