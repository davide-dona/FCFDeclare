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
    min_support=0.3,
    min_confidence=0.9,
    max_cardinality=2,
)
model.constraints[fcf.Constraint('Response', ('Register', 'Close'))]
# Measures(support=1.0, confidence=1.0)
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

Every template with an activation requires its activation to occur, as the table shows. A trace in which the activation does not occur satisfies the constraint **vacuously**, which is the usual reading of Declare: Response$(a, b)$ says nothing about a trace without $a$. `Checker` always applies this reading.

## Mining

The candidates are every unary template on every activity of the log, and every binary template on every ordered pair of distinct activities. Existence, Absence, and Exactly are instantiated for $n = 1, \dots,$ `max_cardinality`. A trace **witnesses** a candidate when the candidate's activation occurs in it and the candidate holds; a template without an activation counts every trace as activated. Each candidate gets two measures, after Maggi, Bose, and van der Aalst (CAiSE 2012):

- **support**: the fraction of the traces of the log that witness it. It says how much of the log the constraint describes.
- **confidence**: the fraction of the traces in which its activation occurs that witness it. It says how reliably the constraint holds when it applies.

A candidate is kept when its support is at least `min_support` and its confidence is at least `min_confidence`. Both thresholds are inclusive, and each filters on its own measure, so changing one never changes what the other means.

Vacuous traces count toward neither measure, so a rare activation cannot make a constraint look strong: if $a$ occurs in a single trace of $N$, Response$(a, b)$ has a support of at most $1 / N$, whatever the other traces do. Conversely, a rule that always holds when it applies has a confidence of 1 however rare its activation. The fraction of the traces that satisfy a constraint, vacuously or not, follows from the two measures and the fraction $p$ of the traces in which the activation occurs: $1 - p\,(1 - \text{confidence})$.

For a template without an activation every trace is activated, so its confidence equals its support, and it is kept when its support reaches both thresholds.

The constraints of a model are ordered by template in the order of the table, then by $n$, then by activity names. This order fixes the columns of `Checker.holds` and the saved file, so equal models save to identical bytes.

### Relation to other Declare miners

The Apriori-based Declare miner of [Maggi, Bose, and van der Aalst (CAiSE 2012)](https://doi.org/10.1007/978-3-642-31095-9_18), which [Declare4Py](https://github.com/ivanDonadello/Declare4Py) implements, takes as binary candidates only the pairs of activities that occur together in enough traces. Templates that hold precisely when two activities do not co-occur, such as Not Responded Existence and Exclusive Choice, are then never candidates. The paper suggests adding negated activities to the itemsets to recover them, but as a way to skip candidates it changes no result. FCFDeclare instead takes every ordered pair of activities, with no co-occurrence requirement, and evaluates all candidates of a template in one vectorized pass over the distinct traces of the log.

The support and confidence above are those of the paper. FCFDeclare has one threshold per measure and no vacuity setting, so no parameter changes what another one counts.

## Saved models

`DeclareModel.save` writes JSON:

```json
{
    "settings": {
        "min_support": 0.3,
        "min_confidence": 0.9,
        "max_cardinality": 2
    },
    "constraints": [
        {
            "template": "Response",
            "activities": ["Register", "Close"],
            "n": 1,
            "support": 1.0,
            "confidence": 1.0
        }
    ]
}
```

`DeclareModel.load` accepts only this format and raises `ValueError` on a missing or extra key, an invalid setting or constraint, a repeated constraint, or a measure outside $[0, 1]$ or a support above its confidence.

## Development

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
```

## License

MIT
