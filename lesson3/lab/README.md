# Lab 3: Model Routing + Agent Workflow Optimization + Agentic Routing

**Course:** ByteByteGo AI Cost Optimization - Nov Cohort | **Instructor**: Jeremy Hintz  
**Estimated time to complete:** 90-120 minutes

## 1. Purpose

In the very first lecture we framed the tokenomics of LLM inference as (number of tokens) x (price per token).

The previous labs explored techniques for optimizing the **number of tokens**: ways we could make each model API request smaller: prompt caching, semantic caching, compressing and compacting the context window, and structuring memory for smart retrieval. This lab approaches the other dimension: **price per token**. The per-token cost of strong models and relatively weaker models varies drastically. In this lab, we’ll explore routing across a strong and weak model, where the strong model costs ~20x the weak one… so routing intelligently is going to be essential for managing our costs.

The topics in this lab will cover:

* **Model routing**: deciding which model answers each request. We’ll step through three variants of increasing sophistication and complexity: **deterministic** (rules in the gateway, embedding routes in the client), a **model cascade** (cheap first, escalate on doubt), and **preference-based** routing with RouteLLM.

* **Agent workflow optimization**: an agent re-reads its whole context on every step. **Planning** buys the expensive model once and executes on cheap ones in fresh contexts. **Memory** optimizes for doing better than straight up dropping facts during compaction, like we saw happen in Lab 2.

* **Agentic routing**: in a multi-agent system the routing decision moves from "which model answers this question" to "which agent handles this sub-task", and each agent brings its own model, tools, and slice of the knowledge base

This lab is intended to build on top of everything we have built so far. Every model call still goes through the LiteLLM mesh you built in Lab 1,with each routing decision traced and instrumented. We will also use Lab 2's index and embedding pipeline over the same Northstar Commerce knowledge base. The embedding model that powered Lab 2's semantic cache will now also be used to map prompts to clusters to make routing decisions.

![Lab 1-3 Cost Optimization Matrix](../../assets/diagrams/lab3_image1.png)

**Every experiment in this lab denotes the accuracy and cost.** While caching and compression return the same answer for less money, routing often returns a different answer. Even if we save some money, a routing decision made without a quality dimension is not sufficient for a production deployment.

## 2. Learning Objectives

By the end of this lab, you will be able to:

1. **Explain** how model routers can help us optimize for both cost and quality of the responses coming from our system  
2. **Run multiple routing schemes:** deterministic, cascade, and preference-based  
3. **Compare** a single tool-calling agent to plan-and-execute, and reason about how each one spends its tokens  
4. **Route** sub-tasks to specialist agents with an LLM supervisor and with a vector supervisor, then re-calibrate a specialist using measured results

## 3. Architecture

![Lab 3 Multi-Tier Routing Architecture](../../assets/diagrams/lab3_image2.png)

The price ladder (USD per million tokens, input / output):

| Tier | Mesh alias | Model | Price |
| :---- | :---- | :---- | :---- |
| weak | mesh-gpt-mini (Lab 1\) | gpt-4o-mini | \$0.15 / \$0.60 |
| mid | mesh-claude-haiku (Lab 1\) | claude-haiku-4-5 | \$1.00 / \$5.00 |
| strong | mesh-claude-sonnet (new) | claude-sonnet-4-6 | \$3.00 / \$15.00 |

Cost in this lab is read from the mesh: LiteLLM returns an x-litellm-response-cost header on every response, the same number Langfuse shows. The prices in config.yaml are only a fallback for models LiteLLM has no price for.

## 4. Prerequisites

Before starting the timed portion, make sure you have:

- **Completed Lab 1 & 2:** LiteLLM proxy from lesson1/lab/llm-mesh-lab. Vector DB scaffold in lesson2/lab/rag-lab/.chroma/ exists with the default chunking. This lab reads from that index.  
- **API Provider Keys:** OPENAI\_API\_KEY serves the weak tier. ANTHROPIC\_API\_KEY serves the medium and strong tiers.  
- **Installed Python 3.10+** and the ability to create a virtual environment. (This lab was verified on Python 3.14 with routellm 0.2.0.)  
- **LiteLLM recent enough for `mesh-auto`.** Lab 3's gateway router uses `auto_router/complexity_router`, which needs LiteLLM **≥ 1.94**. If you installed LiteLLM during Lab 1 weeks ago, upgrade the Lab 1 venv before starting the mesh: `pip install -U 'litellm[proxy]'`.  
- **Langfuse is optional but useful.** If you wired langfuse\_otel in Lab 1, uncomment that callback in mesh\_config.yaml. Every call in this lab is tagged lab3 along with additional metadata for routing-strategy, tier, and agent info.

⚠️ **Spend:** this lab uses a frontier-class model, so it costs slightly more than Labs 1 and 2. One full run of run\_experiment.py should cost about **\$0.50** as of when this lab was written (November 2026). Budget **\$1-2** for the whole lab.

**Mileage will vary.** Models are not perfectly repeatable even at temperature 0. Between two of our own runs the weak tier's score on the 32-question workload moved by a few questions. The trends around the cost structure should be stable, but individual answers may not be.

## 5. A Note on Using AI

Like many (most) things these days, using AI (Claude Code, Cursor, etc) will probably knock this exercise out at break-neck speed. That may or may not be the best way to actually learn something, depending on the person. If you want to use AI to write the code for this lab, go for it! But I encourage you to reason about the code itself and try to understand the ideas. This isn't for a grade or anything, it's purely for you to learn.

## Part 1: Model Routing

### 1.1 Set up the project and start the Lab 3 mesh

```sh
cd lesson3/lab/routing-lab
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

This installs the same retrieval stack as Lab 2 plus routellm.

Stop the Lab 2 proxy if it is still running (and you can stop the lab2-qdrant container; nothing here uses it). In a **second terminal**, start LiteLLM on the **Lab 3** mesh config. It is your Lab 1 config plus a strong-tier alias, Lab 2's embedding alias, and one router alias:

```sh
cd lesson1/lab/llm-mesh-lab
source .venv/bin/activate
pip install -U 'litellm[proxy]'         # mesh-auto needs LiteLLM ≥ 1.94
export OPENAI_API_KEY="sk-..."          
export ANTHROPIC_API_KEY="sk-ant-..."
litellm --config ../../../lesson3/lab/routing-lab/mesh_config.yaml
```

Verify the aliases:

```sh
curl http://localhost:4000/v1/models -H "Authorization: Bearer sk-mesh-lab-1234"
```

**Checkpoint 0:** the list includes mesh-gpt-mini, mesh-claude-haiku, mesh-claude-sonnet, mesh-embed, and mesh-auto. Open mesh\_config.yaml and find mesh-auto. Then open config.yaml and find the tiers: block that maps weak / mid / strong to aliases. 

Leave this terminal running.

### 1.2 Price the ladder

fixtures/queries.json holds 32 queries about Northstar policies: 15 factual lookups (e.g. "How many days do I have to request a refund?") and 17 that need a multi-step derivation from the same underlying policy documents (prorate a refund, count business days, apply a threshold, etc). Each has a regex for the correct final answer. A starter client\_test.py is already in this directory:

```py
from routing import Router, final_line

EASY = "How many days after delivery do I have to request a refund?"
HARD = (
    "I mistyped my password 3 times at 10:00 a.m. and 2 more times at 10:20 a.m. "
    "Is my account locked?"
)

router = Router.from_config("config.yaml")
```

Write code to ask both questions on the weak tier and on the strong tier (router.ask(question, strategy="weak")). For each of the four results print the model that answered, prompt and completion tokens, cost, and the final answer line. Then print how many times more the strong tier cost for the same question.

**Checkpoint 1:** both tiers answer the easy question correctly (30 days), but the strong tier costs roughly 20-30x more. On the hard question the strong tier answers **No** (the lockout rule is five failures within 15 minutes, and these span 20). On our runs the weak tier answered **Yes**. This illustrates the basics of the routing problem: the strong model is wasted on the first question but necessary on the second.

**Think:** Open knowledge\_base/password\_reset.txt in the Lab 2 directory and find the lockout rule. Both models were handed the same retrieved text. What did the weak model get wrong: retrieval, reading, or reasoning? 

### 1.3 Deterministic routing part 1: hard-coding rules in the gateway

The cheapest possible router is a set of rules. LiteLLM ships one: a request for mesh-auto is scored on surface features of the prompt (length, code keywords, reasoning markers, technical terms) and mapped to SIMPLE / MEDIUM / COMPLEX / REASONING. No model is called and nothing is embedded.

```py
from routing import Router

router = Router.from_config("config.yaml")
questions = [
    "How long does expedited shipping take after pickup?",
    "Uptime last month was exactly 99.0% and our subscription is $800 per month. How much service credit should we get?",
    "Think step by step and explain why a 429 is not counted as downtime under the SLA.",
    "Write a Python function that retries a request after a 429 using the Retry-After header.",
]
for q in questions:
    r = router.ask(q, strategy="gateway")
    h = r.calls[0].headers
    print(f"{h.get('x-litellm-complexity-router-tier'):<9} {h.get('x-litellm-complexity-router-score'):>6}  {r.model:<36} {q[:60]}")
```

The last two questions land in MEDIUM and go to Haiku: "step by step" and "Python function" are what the rules look for. The service-credit question lands in SIMPLE and goes to the weak model. Even though it is the one that needs care, it is short, has no code, and uses no reasoning words. The built-in rules measure how complicated a prompt *looks*, but they don’t actually know anything about Northstar or even the policies.

In mesh\_config.yaml, uncomment the custom\_dimensions block under mesh-auto, then **restart LiteLLM**. The block adds one scoring dimension, moneyMath, that fires on dollar amounts, percentages, and words like "credit" and "restocking". Re-run the snippet.

**Checkpoint 2:** after the restart, the service-credit question scores around +0.35, lands in COMPLEX, and is served by anthropic/claude-sonnet-4-6. The shipping question is unchanged. Nothing in the Python client changed: the routing policy lives in the gateway, so one config edit changed routing for every application behind the mesh.

**Think:** Your rule also fires on "What is the restocking fee on an item I opened?", which is a lookup. What does that false positive cost per request? Now imagine maintaining forty such rules. Rules are free to run and auditable in a code review. What is their real cost?

**Think:** The gateway router scores the user's question and ignores the retrieved context in the system prompt. Is that a limitation or a feature for a RAG application, where every prompt is long?

### 1.4 Deterministic routing \#2: embedding routes

Open fixtures/routes.json. It defines five departments (orders, identity, billing, platform, privacy). Each has a handful of example utterances and a tier. A new question is embedded and matched to the route whose closest utterance is nearest. This is Lab 2's semantic cache with one change: the nearest stored text returns a route instead of an answer.

```py
from routing import Router

router = Router.from_config("config.yaml")
for q in [
    "How long does expedited shipping take after pickup?",
    "Uptime last month was exactly 99.0% and our subscription is $800 per month. How much service credit should we get?",
    "I reset my password this morning. Did that also revoke my API keys?",
    "What is the capital of France?",
]:
    d = router.vector_router.route(q)
    print(f"{str(d.route):<9} {d.score:.2f} → {d.tier:<6} | runner-up {d.scores[1][0]} {d.scores[1][1]:.2f} | {q[:60]}")
```

**Checkpoint 3:** shipping queries route to orders (weak) and the service credit to billing (strong). 

**Think:** The tier on each route is still a human's best guess: "billing is money, so since it’s high stakes, send it to the strong model." That sends "On which day are invoices generated?" to the strong model too. A department is a coarse proxy for difficulty. Hold that thought until Part 4, where you can check the guess against data.

### 1.5 Model cascade

Routes and rules make their decision before any model runs, so they can only guess the difficulty. A cascade lets the weak model try, then evaluates a gate as to whether the answer is good enough to keep. This logic is covered in the FrugalGPT paper and is another simple way to route requests. 

```py
from routing import Router, final_line

router = Router.from_config("config.yaml")
for q in [
    "How many days after delivery do I have to request a refund?",
    "I mistyped my password 3 times at 10:00 a.m. and 2 more times at 10:20 a.m. Is my account locked?",
]:
    r = router.ask(q, strategy="cascade")
    print(f"\n{q}\n  {r.reason}")
    for c in r.calls:
        print(f"  {c.purpose:<7} {c.tier:<6} in={c.prompt_tokens:<5} out={c.completion_tokens:<4} ${c.cost_usd:.5f}")
    print(f"  total ${r.cost_usd:.5f}  →  {final_line(r.answer)}")
```

**Checkpoint 4:** the easy question makes two calls (weak answer, weak gate), the gate scores it near 1.0, and the weak answer is kept. The hard question makes three calls: the gate scores the weak answer near 0, and the strong tier answers. Compare that total to what the strong tier alone cost you in 1.2. The escalated question is more expensive than going straight to strong, because we paid for the failed attempt too.

**Think:** A cascade really only works well if escalations are rare. If the weak answer plus the gate cost *w*, the strong answer costs *s*, and a share *p* of traffic escalates, the cascade costs *w + p·s* per question against *s* for all-strong. With your numbers from this step, at what escalation rate does the cascade stop saving money?

Now measure the gate over the whole workload:

```sh
python run_experiment.py --routing --arms B,E
```

**Question:** Count your saves, false alarms, and misses. Raise cascade.threshold to 0.9 and re-run arm E, then lower it to 0.2. Which of the three counts moved, and what happened to the cost?

### 1.6 Preference-based routing with RouteLLM

Rules and routes are both manually configured, and a cascade pays for its decision with extraneous calls. RouteLLM is a more sophisticated design: it uses a classical ML model, trained on tens of thousands of human preference votes between pairs of LLMs, that predicts from the prompt alone **the probability that the strong model's answer would be preferred** over the weak model's. You choose a threshold; prompts at or above it go to the strong model. 

RouteLLM’s out-of-the-box thresholds are calibrated on public chatbot traffic. config.yaml starts with their "about half to strong" value, 0.11593. See what it does on a sample of *your* traffic: fixtures/traffic\_sample.json holds 40 unlabeled support messages that are not in the graded workload.

```sh
python calibrate.py
```

The first run downloads the router checkpoint from Hugging Face, and importing RouteLLM can be slow, so expect up to a minute before the first score.

**Checkpoint 5:** the script prints the distribution of P(strong wins) over the sample, the highest- and lowest-scoring messages, and a threshold for each target share. On our run every score fell between about 0.04 and 0.35, and the published threshold sent about 60% of the sample to the strong model. The threshold is not portable: it was only intended for the traffic it was calibrated on.

Now calibrate to a budget instead of a guess. Calibration needs no labels, only a traffic sample and a decision about how much strong-model usage you can afford:

```sh
python calibrate.py --strong 0.30
```

Copy the printed value into preference.threshold in config.yaml, then compare the router against all-weak on the graded workload:

```sh
python run_experiment.py --routing --arms B,F
```

**Think:** Look at the highest- and lowest-scoring messages from calibrate.py. RouteLLM has never seen a Northstar policy and it doesn’t read the retrieved context. What is it reacting to instead?

## Part 2: Agent Workflow Optimization 

A question takes one model call. A support can require a heck of a lot more than that: answering "how much will my refund be?" means looking up the order, looking up the policy, and doing arithmetic. That is an agent: a model in a loop with memory and tools. agents.py gives the model six read-only tools (fixtures/tools.json): search\_kb over the Lab 2 index, plus get\_order, get\_account, get\_uptime, get\_api\_usage, and get\_login\_status over synthetic records in fixtures/crm.json.

The loop presents a cost problem. Open run\_agent in agents.py and find the messages list. Note how as it stands, nothing is ever removed from it. Every call re-sends the system prompt, all six tool schemas, and every tool result so far.

### 2.1 Simple agent

fixtures/tickets.json holds four tickets: two with a single issue and two with three issues each. Run the third one (a refund, a service credit, and a rate-limit question in one message) through a single strong agent:

```py
import json
from agents import Desk

desk = Desk.from_config("config.yaml")
ticket = json.load(open("fixtures/tickets.json"))["tickets"][2]["text"]

r = desk.handle(ticket, mode="react")
for i, c in enumerate(r.calls, 1):
    print(i, c.purpose, c.tier, c.prompt_tokens, c.completion_tokens, f"${c.cost_usd:.4f}", [t["name"] for t in c.tool_calls])
print(f"total: {r.llm_calls} calls, {r.prompt_tokens} prompt tokens, ${r.cost_usd:.4f}")
print(r.answer)
```

**Checkpoint 6:** the agent resolves all three issues. Look at prompt\_tokens from one call to the next: each call is larger than the last, because it carries every tool result that came before. On our runs the strong model finished in two or three calls, batching several tool calls into each turn, and its last call was more than twice the size of its first. Every one of those tokens was billed at the strong tier's price, including the ones spent reading a refund policy while answering a rate-limit question.

### 2.2 Plan first, then execute 

Plan-and-execute splits the work in two. A strong model reads the ticket once and writes a plan: a short list of self-contained steps. Each step then runs on a **cheap** model in a **fresh** context that holds only that step. A cheap model stitches the results into one reply.

```py
r = desk.handle(ticket, mode="plan")
for s in r.plan:
    print("step", s.id, s.depends_on, s.task[:100])
for i, c in enumerate(r.calls, 1):
    print(i, c.purpose, c.tier, c.prompt_tokens, c.completion_tokens, f"${c.cost_usd:.4f}", [t["name"] for t in c.tool_calls])
print(f"total: {r.llm_calls} calls, {r.prompt_tokens} prompt tokens, ${r.cost_usd:.4f}", r.cost_by_tier())
print(r.answer)
```

**Checkpoint 7:** Look at cost\_by\_tier(): on our runs the single planner call was the majority of the whole ticket's cost, mostly because of the plan's *output* tokens at the strong tier's output price.

Now read the reply against the facts. The correct answers are a **\$200** refund (the buyer is a Pro subscriber, so the 15% restocking fee is waived), a **\$180** service credit (97.2% uptime is below 99.0%, so 15% of \$1,200), and a limit of **300** requests per minute exceeded by **150**.

**Think:** Run the first ticket (\["tickets"\]\[0\], a single refund question) in both modes. The planner call is pure overhead when there is nothing to split. What cheap check could decide whether a ticket needs a planner at all?

### 2.3 Memory

Lab 2 ended on a question: compaction would drop the turn where the customer said the order number. Here is that failure, and its fix.

fixtures/session.json is a ten-turn support chat adapted from Lab 2's conversation fixture. In turn 1 the customer gives an account ID and an order number. Turns 7 and 10 are **callbacks**: they can only be answered from something said much earlier. Run it three ways:

```py
import json, re
from agents import Session
from routing import load_config

turns = json.load(open("fixtures/session.json"))["turns"]
for mode in ("full", "compact", "memory"):
    session = Session(load_config("config.yaml"), mode=mode)
    print(f"\n=== {mode} ===")
    for i, t in enumerate(turns, 1):
        turn = session.send(t["text"])
        mark = ""
        if "expect" in t:
            mark = "PASS" if all(re.search(p, turn.answer, re.I) for p in t["expect"]) else "FAIL"
        print(f"{i:>2} prompt={turn.calls[0].prompt_tokens:<5} history_kept={turn.history_messages_sent:<2} calls={len(turn.calls)} {mark:<4} {turn.answer[:70]}")
        if turn.recalled:
            print("      recalled:", turn.recalled)
```

- full sends the whole conversation every turn.  
- memory is compaction plus a small fact store. After each turn one extra cheap call extracts durable facts (identifiers, amounts quoted, decisions). Before each turn the facts most similar to the new message are recalled, using the same MiniLM embedder, and placed in the system prompt.

## Part 3: Agentic Routing in a Multi-Agent System

Part 2 left a problem: cheap workers are cheap everywhere, including on the steps that need a better model. Part 1 already solved that problem for individual questions, so now we will take the same principles and extend them to multiple sub-tasks of varying complexity.

Open fixtures/routes.json. In Part 1 you used each route's tier. Each route also has tools and sources. Together those three fields define a **specialist agent**: the billing specialist runs on the strong tier, carries three tools instead of six, and searches two documents instead of eight. The supervisor's job is to decide which specialist gets each piece of a ticket. That decision is a routing decision.

### 3.1 Hierarchical agent coordination with supervisor

```py
import json
from agents import Desk

desk = Desk.from_config("config.yaml")
ticket = json.load(open("fixtures/tickets.json"))["tickets"][2]["text"]

r = desk.handle(ticket, mode="team", supervisor="llm")
for note in r.notes:
    print("note:", note)
for run in r.runs:
    print(f"{run.agent:<9} [{run.tier:<6}] calls={len(run.calls)} ${run.cost_usd:.4f} tools={[t.split('(')[0] for t in run.tool_log]}")
print(f"total: {r.llm_calls} calls, ${r.cost_usd:.4f}", r.cost_by_tier())
print(r.answer)
```

The planner from 2.2 now also names an agent for every step.

**Checkpoint 8:** three specialists ran: orders on the weak tier, billing on the strong tier, platform on the mid tier. Each used only its own tools. Check the four facts again: the service credit, which the cheap worker tended to miss in 2.2, is now computed by a strong model, while the refund lookup still runs on the cheapest one.

**Think:** Compare the three totals for this ticket: react (2.1), plan (2.2), team. The team probably did not come out much cheaper than the single strong agent, and it is certainly more expensive than plan. Where did the money go? Use cost\_by\_tier() and the per-agent lines. A multi-agent system is not cheaper by construction. It is cheaper only when work lands on cheap agents.

### 3.2 Agentic Routing without an LLM supervisor

The LLM supervisor is a strong-model call whose output is mostly a routing table. You already own a router that makes that decision without a model:

```py
r = desk.handle(ticket, mode="team", supervisor="vector")
for note in r.notes:
    print("note:", note)
for run in r.runs:
    print(f"{run.agent:<9} [{run.tier:<6}] calls={len(run.calls)} ${run.cost_usd:.4f} tools={[t.split('(')[0] for t in run.tool_log]}")
print(f"total: {r.llm_calls} calls, ${r.cost_usd:.4f}", r.cost_by_tier())
```

The vector supervisor routes each sentence of the ticket with Part 1's embedding router. Every department that wins a sentence gets the whole ticket and is told to handle only its own part. No planner call is made.

**Checkpoint 9:** the same three specialists run (orders / billing / platform), but there is no planner call in the cost breakdown. On our runs the weak orders specialist was more likely to miss the Pro restocking-fee waiver when handed the whole multi-issue ticket instead of a rewritten single-step task.

**Question:** The vector-based router saved the planner call. What did it trade off? Compare the task each specialist received in 3.1 (the planner's rewritten, self-contained instruction) with 3.2 (the entire ticket plus "handle only your part"). Look at the specialists' prompt tokens, and at whether the weak orders specialist still gets the refund right. Run the single-issue ticket (\["tickets"\]\[0\]) both ways: for which kind of ticket is the vector supervisor the clear winner?

### 3.3 Re-tier with data

Every tier in routes.json was a guess. You now have a way to test one. In fixtures/routes.json, change the billing route's tier from strong to mid, then run the team over all four tickets:

```sh
python run_experiment.py --agents --arms react,team-llm
```

**Checkpoint 10:** the team's cost drops substantially against react, because the most expensive specialist moved down a rung. Read the facts column and any MISSED: lines before you celebrate. On our run this change cut the team's cost by roughly 40% and lost one of the nine facts.

Put billing back to strong before Part 4, or keep mid and note it in your table. Either is fine as long as you know which one you measured.

## Part 4: The Cost Experiment

You now have every lever. Act like a cost engineer: measure them side by side, with quality in the same table.

### 4.1 Run the experiment

run\_experiment.py runs three workloads through the mesh using the numbers in your config.yaml and routes.json (your calibrated RouteLLM threshold, your cascade threshold, your specialist tiers, and whatever gateway rule your running mesh has loaded).

```sh
python run_experiment.py
```

A full run makes a few hundred model calls and takes roughly ten minutes. To re-run one workload or a few arms: --routing, --agents, or --memory, optionally with --arms, for example python run\_experiment.py --routing --arms A,E.

**Workload 1, model routing.** The 32 graded questions, each with the same retrieval. Only the model reading the context changes.

| Arm | Strategy | Decides | Extra cost of deciding |
| :---- | :---- | :---- | :---- |
| A | all-strong | never | none |
| B | all-weak | never | none |
| C | gateway rules (mesh-auto) | before the call | none |
| D | embedding routes | before the call | a local embedding |
| E | cascade (weak → gate → strong) | after the weak call | a gate call on every question |
| F | RouteLLM | before the call | one small embedding call |

**Workload 2, agent architectures.** The four tickets, graded on nine facts: react, plan, team-llm, team-vector.

**Workload 3, memory.** The ten-turn chat: full, compact, memory.

The routing workload prints a grid with one row per question and one column per arm, showing which tier answered and whether it was right. Per-item results, including every answer, are saved to results.json.

If you have Langfuse from Lab 1, filter to tag lab3 and then arm-A versus arm-E. Arm E's traces show the lab3-gate generations that arm A does not have.

---

## Advanced Exercise: Train a Custom Router for RouteLLM

In Part 1.6 you downloaded a router and turned the most basic dial available to us: the threshold for choosing one model over another. The model behind that dial was trained on Chatbot Arena votes between models such as GPT-4 and Mixtral, on public chatbot traffic. It has never seen a Northstar policy, a Northstar question, or either of the two models your mesh serves. You calibrated its threshold to your traffic. That’s limiting, because you couldn’t influence what it believes is hard relative to your data in particular.

This exercise goes inside that black box. You implement RouteLLM's matrix-factorization model and its training loop, train it on **battles between your own two tiers on Northstar questions**, save it in RouteLLM's checkpoint format, and load it back into arm F. Writing and testing the code is offline. Training needs the mesh for one embedding request (well under a cent). Routing with the result costs about what arm F cost you in Part 1.

![Lab 3 PyTorch Matrix Factorization Router](../../assets/diagrams/lab3_image3.png)

Matrix Factorization is a type of classical ML model that is commonly used for recommender systems (think: “users like you also purchased these items” type use cases). If you want to learn about how a matrix factorization model really works, here is a good YouTube video for you to check out: [https\://www.youtube.com/watch?v=ZspR5PZemcs](https://www.youtube.com/watch?v=ZspR5PZemcs) 

This lab assumes we have labeled training data to build the matrix-factorization on top of. mf-router-lab/fixtures/battles.json holds ~300 Northstar questions that are not in the graded workload. Each was answered by the weak and the strong tier through the Part 1 pipeline, and a judge decided whether the weak answer reached the strong answer's conclusion. The strong tier "won" 72 of them (23%); the rest are ties. Producing that file cost about \$1.50, almost all of it coming from strong-tier answers and verdicts. collect\_battles.py is the script that made it, and if you want to go really deep you can read it or run it to make more.

### The mental model

- A **battle** is one query, two models, and a judge’s verdict: model\_a won, model\_b won, or tie.  
- The model scores a (model, prompt) pair: score \= w · (normalize(P\[model\]) ⊙ W·e(prompt)). e(prompt) is the prompt's mesh-embed vector, computed once and frozen. W projects it from 1536 numbers to 128. P\[model\] is one learned vector per model. w turns the element-wise product into one number.  
- **P(strong wins)** is sigmoid(score(strong) − score(weak)). A router only ever compares two models on the same prompt.  
- **Training** turns each battle into a (winner, loser) pair and pushes score(winner) − score(loser) up. Gaussian noise on the embeddings is the regularizer.  
- The **checkpoint** is three tensors with RouteLLM's names and shapes. Your strong and weak tiers reuse the two rows of the model table that RouteLLM scores by default, so RouteLLM loads your file with no code changes.

Edit only lesson3/lab/mf-router-lab/mf\_router.py. Implement the seven functions marked TODO, in order:

| Function | Purpose |
| :---- | :---- |
| MFModel.score | How well a model is expected to handle a prompt, as a logit |
| MFModel.win\_logit | score(winner) − score(loser) |
| battle\_pairs | Judged battles → (winner, loser) training pairs; decides what a tie means |
| train\_epoch | Shuffle, add noise, binary cross-entropy against "the winner won", step |
| strong\_win\_rate | P(strong wins) for each prompt: the number the router returns |
| auc | Does the router rank prompts correctly? No threshold involved |
| budget\_table | What each strong-tier budget catches, and what it leaves on the weak tier |

Everything else is provided: the model's constructor, the multi-epoch fit loop, threshold calibration (the same function as Part 1.6), checkpoint export, and train.py. Each function has a detailed docstring spelling out the contract. The seven functions come to about seventy lines of PyTorch and plain Python; you do not need a GPU.

### Workflow

```sh
cd lesson3/lab/mf-router-lab
source ../routing-lab/.venv/bin/activate    # the hands-on lab's environment already has PyTorch
pip install -r requirements.txt             # adds pytest

# 1. Implement the TODOs in mf_router.py, checking your work as you go:
pytest -v

# 2. Once green, train on the real battles (the Lab 3 mesh must be running):
python train.py
```

train.py embeds the 307 prompts through mesh-embed (cached after the first run), scores RouteLLM's published router on them as a baseline, cross-validates your model five ways so that every score comes from a model that never trained on that prompt, then trains a final model on everything and saves it to checkpoints/northstar-mf/. It takes about fifteen seconds on a laptop.

On our runs the published router's AUC on these battles was about **0.60**, and a router trained here scored **0.68 to 0.73** depending on the seed. At a 30% strong-tier budget, a router that knows nothing catches 30% of the strong wins; the published router caught 44% and ours 56%. Training is seeded, but floating-point results differ between machines, so your numbers will be close to these and not identical.

**Think:** 0.5 is a coin flip and 1.0 is perfect. The published router, trained on tens of thousands of battles, manages 0.60 on Northstar questions. Yours, trained on about 250, beats it. What does that say about where a router's value comes from? And why is yours still nowhere near 1.0? (Open battles.json and read ten strong wins. How many could you have predicted from the question alone?)

### Break it on purpose

Each of these is one command. Compare the AUC line and the per-fold numbers with your first run.

```sh
python train.py --lr 3e-4 --alpha 0.1    # the hyperparameters in RouteLLM's training script
python train.py --ties drop              # what RouteLLM's training script does with ties
python train.py --init published         # start from RouteLLM's weights instead of random ones
python train.py --curve                  # train on 25 / 50 / 75 / 100% of the battles
```

**Question:** With RouteLLM's own learning rate and noise level, our router scored 0.51 to 0.56 depending on the seed, below the published one. The embeddings have length 1, so each of their 1536 numbers is around 0.025. What does adding noise with standard deviation 0.1 to every one of them do to a prompt? Why would that be a sensible setting for 100,000 battles and a bad one for 250?

### Route with your model

Your checkpoint is a RouteLLM checkpoint. In ../routing-lab/config.yaml, point the preference strategy at it:

```
preference:
  checkpoint: ../mf-router-lab/checkpoints/northstar-mf
```

A threshold belongs to the model that produced the scores, so the one in config.yaml now means nothing. Recalibrate on the traffic sample, copy the printed value into preference.threshold, and re-run arm F:

```sh
cd ../routing-lab
python calibrate.py --strong 0.30
python run_experiment.py --routing --arms B,F
```

calibrate.py and the experiment both print which weights they loaded. Look at the highest- and lowest-scoring messages from calibrate.py and compare them with what the published router ranked highest in Part 1.6. On our run the top of the list was prorated refunds and service-credit calculations, the question types the weak tier got wrong most often in the battles.

On our run, with both routers calibrated to 30% of the traffic sample, the published router sent 53% of the graded questions to the strong tier and answered 31 of 32; ours sent 34% and answered 30 of 32, for about 70% of the published router's cost. When we ranked the 32 questions by each router's score and compared them at equal numbers of strong calls, the two caught the same number of the weak tier's five mistakes at almost every budget.

**Question:** The battles say your router is better than the published one. The graded workload says they are about the same. Which do you believe, and why? Count the weak tier's wrong answers in your arm B: that is the number of questions on which two routers can differ at all. How large would the graded workload have to be to show a difference of 0.60 versus 0.70 in AUC?

Score a few messages directly to see what changed. Run this from routing-lab with your checkpoint, then again with checkpoint: null:

```py
from routing import Router

router = Router.from_config("config.yaml")
for text in [
    "How long is a password reset link valid?",
    "Uptime last month was 98.7% and our subscription is $1,200 per month. How much service credit do we get?",
    "Hi, is anyone there?",
    "What is the capital of France?",
]:
    print(f"{router.preference.win_rate(text):.3f}  {text}")
```

On our run the published router scored the first two almost identically (0.14 each). Ours scored the reset-link lookup 0.004 and the service-credit calculation 0.62.

**Think:** The battles contain no small talk and nothing off-topic. Your router gave the last two messages a low, confident score anyway (0.02 and 0.04 on our run), so they go to the weak tier. Nothing in its training data says that is right. What did it base those numbers on? What would you add to battles.json before shipping this, and what does a router trained on public chatbot traffic have that yours lacks?

### Definition of done

1. All tests in test\_mf\_router.py pass (pytest -v), including the one that loads your checkpoint with RouteLLM's own MatrixFactorizationRouter and gets the same number you computed.  
2. python train.py prints a held-out AUC for your router that is above the published router's on the same battles, and saves checkpoints/northstar-mf/model.safetensors.  
3. python calibrate.py in routing-lab reports that it loaded your checkpoint, and arm F runs with it.  
4. You can explain why your router's AUC is higher than the published ones on the battles and why the 32 graded questions cannot confirm it.

If you run out of gas or just want to run a completed solution, you can find the solution in reference\_solution.py (MF\_ROUTER\_IMPL=reference\_solution python train.py).

### References

- The exercise: [mf-router-lab/README.md](mf-router-lab/README.md)  
- Hands-on router (where preference.checkpoint is read): [routing-lab/routing.py](routing-lab/routing.py)  
- RouteLLM's matrix-factorization model and training script: [https\://github.com/lm-sys/RouteLLM/tree/main/routellm/routers/matrix\_factorization](https://github.com/lm-sys/RouteLLM/tree/main/routellm/routers/matrix_factorization)  
- The public battles RouteLLM trained on: [https\://huggingface.co/datasets/lmsys/lmsys-arena-human-preference-55k](https://huggingface.co/datasets/lmsys/lmsys-arena-human-preference-55k)

## References

- Lab 1 mesh (required): [../../lesson1/lab/exercise\_overview.md](../../lesson1/lab/exercise_overview.md)  
- Lab 2 index and pipeline (required): [../../lesson2/lab/exercise\_overview.md](../../lesson2/lab/exercise_overview.md)  
- FrugalGPT (LLM cascades): [https\://arxiv.org/abs/2305.05176](https://arxiv.org/abs/2305.05176)  
- RouteLLM paper (learning to route from preference data): [https\://arxiv.org/abs/2406.18665](https://arxiv.org/abs/2406.18665)  
- RouteLLM code and published thresholds: [https\://github.com/lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM)  
- RouteLLM's matrix-factorization training script (the advanced exercise rebuilds it): [https\://github.com/lm-sys/RouteLLM/blob/main/routellm/routers/matrix\_factorization/train\_matrix\_factorization.py](https://github.com/lm-sys/RouteLLM/blob/main/routellm/routers/matrix_factorization/train_matrix_factorization.py)  
- RouteLLM matrix-factorization checkpoint used here: [https\://huggingface.co/routellm/mf\_gpt4\_augmented](https://huggingface.co/routellm/mf_gpt4_augmented)  
- LiteLLM auto routing (the gateway's complexity router): [https\://docs.litellm.ai/docs/proxy/auto\_routing](https://docs.litellm.ai/docs/proxy/auto_routing)  
- LiteLLM Proxy docs: [https\://docs.litellm.ai/docs/simple\_proxy](https://docs.litellm.ai/docs/simple_proxy)  
- Semantic Router (embedding routes as a library): [https\://github.com/aurelio-labs/semantic-router](https://github.com/aurelio-labs/semantic-router)  
- Anthropic, "Building effective agents" (routing, orchestrator-workers): [https\://www.anthropic.com/engineering/building-effective-agents](https://www.anthropic.com/engineering/building-effective-agents)  
- Langfuse ↔ LiteLLM (if you left it on from Lab 1): [https\://langfuse.com/integrations/gateways/litellm](https://langfuse.com/integrations/gateways/litellm)

[image2]: <data:image/png;base64,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>

