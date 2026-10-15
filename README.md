# FCFDeclare

FCFDeclare (Fast Control-Flow Declare) mines Declare models from event logs and checks traces against them. It works on the control flow alone: a trace is a sequence of activity names, and every template is evaluated for a whole batch of traces at once with numpy. 

Its miner derives from [*Efficient Discovery of Understandable Declarative Process Models from Event Logs*](https://doi.org/10.1007/978-3-642-31095-9_18) (Maggi, Bose, and van der Aalst, CAiSE 2012).

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

> [!WARNING]
> A `str` is itself a sequence of one-character strings, so `'abc'` is read as the trace of activities `a`, `b`, `c`.

> [!TIP]
> Each call to `check` or `holds` costs a few numpy operations per template whatever the batch size, so pass all traces in one call rather than one trace per call. Duplicate traces in a batch are evaluated once.

[`examples/mining_and_fitness.ipynb`](examples/mining_and_fitness.ipynb) walks through the whole flow on a small CSV log: loading the traces, mining a model, scoring the fitness of the log and of new traces, and finding the constraints a trace violates.

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
| Responded Existence$(a, b)$ | if $a$ occurs, so does $b$ | $a$ |
| Not Responded Existence$(a, b)$ | if $a$ occurs, $b$ does not | $a$ |
| Response$(a, b)$ | a $b$ follows every $a$ | $a$ |
| Precedence$(a, b)$ | an $a$ precedes every $b$ | $b$ |
| Not Response$(a, b)$ | no $b$ follows any $a$ | $a$ |
| Not Precedence$(a, b)$ | no $a$ precedes any $b$ | $b$ |
| Chain Response$(a, b)$ | a $b$ immediately follows every $a$ | $a$ |
| Chain Precedence$(a, b)$ | an $a$ immediately precedes every $b$ | $b$ |
| Not Chain Response$(a, b)$ | no $b$ immediately follows any $a$ | $a$ |
| Not Chain Precedence$(a, b)$ | no $a$ immediately precedes any $b$ | $b$ |
| Alternate Response$(a, b)$ | a $b$ follows every $a$ before the next $a$ | $a$ |
| Alternate Precedence$(a, b)$ | an $a$ precedes every $b$ since the previous $b$ | $b$ |

A trace in which a constraint's activation does not occur satisfies it **vacuously**, the usual reading of Declare: Response$(a, b)$ says nothing about a trace without $a$.

## Mining

`mine` instantiates the templates as candidate constraints and keeps those that the log supports.

- **Candidates.** Every unary template on every activity of the log, and every binary template on every ordered pair of distinct activities.
- **Non-vacuous satisfaction.** A candidate is non-vacuously satisfied in a trace when its activation occurs in it and the candidate holds; a template without an activation is non-vacuously satisfied wherever it holds. The measures count traces, not events: a trace in which the activation occurs several times counts once.

A candidate is kept when its support reaches `min_support` and its confidence reaches `min_confidence`. Both thresholds are inclusive, and each filters on its own measure, so changing one never changes what the other means. The constraints of a model are ordered by template in the order of the table, then by $n$, then by activity names. This order fixes the columns of `Checker.holds` and the saved file, so equal models save to identical bytes.

### Support

The **support** of a candidate is the fraction of traces in which it is non-vacuously satisfied. It says how much of the log the candidate describes, and `min_support` sets how much is enough.

Support never exceeds the support of the activation, the fraction of traces in which the activation occurs: if $a$ occurs in one trace out of 100, Response$(a, b)$ has a support of at most 0.01. A rare activation therefore cannot make a candidate look strong, but a high `min_support` also drops every rule on a rare activity, however reliable.

The paper's non-vacuous satisfaction also requires that no stronger constraint holds in the trace. FCFDeclare does not, so a trace that satisfies Chain Response$(a, b)$ also counts toward Response$(a, b)$.

### Confidence

The **confidence** of a candidate is the ratio between its support and the support of its activation, which the paper calls the antecedent. Of the traces in which the activation occurs, it is the fraction in which the candidate holds. It says how reliably the candidate holds when it applies, however often it applies, and `min_confidence` sets how many deviating traces are tolerated: 1 tolerates none.

A low `min_support` with a high `min_confidence` keeps reliable rules on rare activities, while `min_support` still bounds how rare their activation can be. For a template without an activation every trace counts, so its confidence equals its support.

### Cardinality

Existence, Absence, and Exactly are the **cardinality** templates: they require $a$ to occur at least $n$, fewer than $n$, and exactly $n$ times respectively. `max_cardinality` sets the counts they are instantiated for, $n = 1, \dots,$ `max_cardinality`, and the other templates have $n = 1$. Absence$(a, 1)$ says that $a$ never occurs and Absence$(a, 2)$ that it occurs at most once, so `max_cardinality=1` can express the first but not the second.

## Checking

`Checker` scores a batch of traces against a model. A trace satisfies a constraint when the condition of the table holds, vacuously or not, so unlike the measures of mining, checking counts every trace without the activation as satisfying the constraint. An activity that the model does not name matches no constraint, but still counts as an event: it can be the first or last event of a trace, and it separates two events for the Chain templates.

- `check` returns, for each trace, `satisfied`, the number of constraints it satisfies; `share`, the fraction of the constraints it satisfies, a fitness in $[0, 1]$; and `full`, 1 when it satisfies every constraint and 0 otherwise. For a model without constraints, both scores are 0.
- `holds` returns a bool matrix of shape $[B, K]$, whether each of the $B$ traces satisfies each of the $K$ constraints, in the order of `model.constraints`. It shows which constraints a trace violates.

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

## License

[MIT](LICENSE)
