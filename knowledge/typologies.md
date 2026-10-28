# GraphGuard typology knowledge base (starter corpus)

> These entries are short PARAPHRASED summaries of widely documented money-laundering typologies and of the
> red-flag categories in FATF's guidance on virtual assets (FATF, "Virtual Assets: Red Flag Indicators of Money
> Laundering and Terrorist Financing", 2020). They are NOT verbatim FATF text. For the final evaluation, drop the
> official FATF PDF(s) and any regulator definitions into this folder (`*.pdf`, `*.txt`, `*.md`); they are indexed
> automatically alongside this file and cited by filename + chunk id.

## [TYP-FANOUT] Fan-out / distribution (dispersal) pattern
A single wallet or transaction that suddenly disperses value to many recipient addresses in a short period is a
recognised layering behaviour. Splitting a large balance into many smaller transfers lowers the visibility of any
single movement and is often the first step of structuring or peel-chain style laundering. Red-flag context: one
source with an unusually high out-degree relative to its in-degree, many low-value outputs, recipients that are
new or have no prior history.

## [TYP-PEEL] Peel chain
A long chain in which a large amount is moved through successive addresses, at each hop "peeling off" a small
portion to a new address and passing the remainder onward. It is used to layer proceeds and to frustrate tracing.
Graph signature: a path of transactions with one large continuing output and one small side output at each hop.

## [TYP-CYCLE] Layering through rapid cyclical fund movement (round-tripping)
Funds that return to, or circulate within, the same small set of addresses (A to B to C and back to A) create the
appearance of legitimate activity or obscure the origin of value. Round-tripping is a classic layering technique
and is also the core signature of wash trading. Graph signature: directed cycles or strongly connected components
of small size, reciprocal transfers between the same counterparties, repeated movement in a short time window.

## [TYP-WASH] Wash trading / closed-cluster collusion
Colluding counterparties trade among themselves to fabricate volume or to move value while appearing to have
economic purpose. Detection literature identifies closed clusters of accounts that trade almost only with each
other. Graph signature: dense, closed clusters with cyclic flows and few connections to the wider network.

## [TYP-BRIDGE] Community-bridging / cross-cluster hopping
Illicit flows often pass through intermediary addresses that connect otherwise separate groups of wallets, e.g. to
move funds from a compromised or illicit cluster towards exchanges or mixing services. A transaction whose
counterparties belong predominantly to different communities than its own is an unusual structural position that
warrants review. Related FATF red-flag themes: rapid movement of funds between multiple service providers or
wallets with no clear business rationale, and use of intermediaries to break the link between origin and
destination.

## [TYP-MIXER] Mixing / tumbling services
Mixers and tumblers pool funds from many users and redistribute them, breaking the on-chain link between source
and destination. Transactions that receive from many unrelated inputs and pay out to many unrelated outputs, or
that interact with known mixing services, are a recognised red flag for concealing the origin of funds.

## [TYP-STRUCT] Structuring (smurfing)
Breaking a larger amount into multiple smaller transactions to remain below reporting or monitoring thresholds.
FATF red-flag guidance lists many small-value transactions, especially in rapid succession or just below a
threshold, as an indicator of possible structuring.

## [TYP-VELOCITY] Unusual transaction velocity and rapid onward movement
Funds received and forwarded almost immediately, with little dwell time and no evident economic purpose, are a
recognised indicator of layering. Activity that is markedly more intense than that of comparable addresses at the
same time is a supporting factor, not proof on its own.

## [REG-SAR] Suspicious activity reporting principle
Institutions are expected to report suspicion of money laundering, associated predicate offences or terrorist
financing to the financial intelligence unit. Red-flag indicators are prompts for further review; no single
indicator is conclusive, and a determination requires context and human judgement. Automated scores should support,
not replace, analyst decision-making.

## [REG-LIMITS] Limits of on-chain evidence
On-chain analysis observes pseudonymous addresses and transfer patterns, not real-world identities or intent.
Structural patterns such as fan-out, cycling or bridging are indicators that increase risk; they can also arise from
legitimate services such as exchanges, payment processors and custodial wallets.
