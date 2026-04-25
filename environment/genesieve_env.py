# genesieve_env.py (COLAB SAFE + STRICT VERSION)

from uuid import uuid4
import json, os, random

# ---------------- CONFIG ---------------- #

BUDGET = 15
MAX_GENES_SHOWN = 20
MIN_VALID_VISIBLE = 3

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data")
)
ORGANISMS = ["ecoli", "saureus", "mtb"]

HIDDEN_GENE_FIELDS = {"binding_compounds", "has_human_homolog", "is_valid_target", "essential"}
PRIOR_NOISE = 0.20

# ---------------- OBS ---------------- #

class GenesieveObservation:
    def __init__(self, organism, budget_remaining, genes_available, last_result, done, reward):
        self.organism = organism
        self.budget_remaining = budget_remaining
        self.genes_available = genes_available
        self.last_result = last_result
        self.done = done
        self.reward = reward


# ---------------- ENV ---------------- #

class GenesieveEnvironment:
    def __init__(self):
        self.env_id = str(uuid4())
        self._rng = random.Random()
        self._state = None
        self._gene_db = {}
        self._load_databases()

    # ---------------- RESET ---------------- #

    def reset(self, seed=None):
        if seed is not None:
            self._rng = random.Random(seed)

        key = self._rng.choice(ORGANISMS)

        if key not in self._gene_db:
            raise ValueError(f"Organism {key} not loaded")

        db = self._gene_db[key]

        if "genes" not in db:
            raise ValueError(f"Invalid dataset: missing 'genes' field")

        all_genes = db["genes"]

        if not all_genes:
            raise ValueError("Dataset contains zero genes")

        lookup = {g["gene_name"]: g for g in all_genes}

        valid = [g for g in all_genes if g.get("is_valid_target")]
        invalid = [g for g in all_genes if not g.get("is_valid_target")]

        if len(valid) == 0:
            raise ValueError("No valid targets in dataset")

        chosen_valid = self._rng.sample(valid, min(MIN_VALID_VISIBLE, len(valid)))
        chosen_invalid = self._rng.sample(invalid, min(MAX_GENES_SHOWN - len(chosen_valid), len(invalid)))

        remaining = [g for g in all_genes if g not in chosen_valid + chosen_invalid]

        needed = MAX_GENES_SHOWN - len(chosen_valid) - len(chosen_invalid)
        extra = self._rng.sample(remaining, min(len(remaining), max(0, needed)))

        visible = chosen_valid + chosen_invalid + extra
        self._rng.shuffle(visible)

        self._state = {
            "organism": db.get("display_name", key),
            "visible_genes": visible,
            "all_genes": lookup,
            "budget": BUDGET,
            "done": False,
            "history": [],
            "step_count": 0,
        }

        return self._build_obs(0.0)

    # ---------------- STEP ---------------- #

    def step(self, action):
        # ---------- STATE SAFETY ---------- #
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self._state.get("done"):
            return self._build_obs(-0.5)

        # ---------- PARSE ACTION ---------- #
        if isinstance(action, dict):
            tool = action.get("tool")
            args = action.get("args", {}) or {}
        else:
            tool = getattr(action, "tool", None)
            args = getattr(action, "args", {}) or {}

        gene_name = args.get("gene_name")

        # ---------- HARD VALIDATION ---------- #
        if tool is None:
            return self._build_obs(-0.3)

        if gene_name is None:
            return self._build_obs(-0.4)

        g = self._state["all_genes"].get(gene_name)

        if g is None:
            return self._build_obs(-0.4)

        # ---------- SAFE FROM HERE ---------- #
        reward = 0.0
        result = None

        valid_tools = {
            "inspect_gene",
            "check_human_homolog",
            "test_binding",
            "submit_target",
        }

        if tool not in valid_tools:
            return self._build_obs(-0.3)

        if tool == "inspect_gene":
            result = g["essential"]
            reward = 0.5 if result else -0.2

        elif tool == "check_human_homolog":
            result = not g["has_human_homolog"]
            reward = 0.3 if result else -0.1

        elif tool == "test_binding":
            result = len(g["binding_compounds"]) > 0
            reward = 0.35 if result else -0.1

        elif tool == "submit_target":
            self._state["done"] = True
            result = g["is_valid_target"]
            reward = 3.0 if result else -1.5

        # ---------- UPDATE STATE ---------- #
        self._state["budget"] -= 1
        self._state["step_count"] += 1

        if self._state["budget"] <= 0 and not self._state["done"]:
            self._state["done"] = True
            reward -= 0.5

        self._state["history"].append({
            "tool": tool,
            "gene": gene_name,
            "result": result,
        })

        return self._build_obs(reward)

    # ---------------- OBS ---------------- #

    def _build_obs(self, reward):
        return GenesieveObservation(
            organism=self._state["organism"],
            budget_remaining=self._state["budget"],
            genes_available=self._prepare_visible(self._state["visible_genes"]),
            last_result=None,
            done=self._state["done"],
            reward=reward,
        )

    # ---------------- PRIORS ---------------- #

    def _prepare_visible(self, genes):
        result = []

        for g in genes:
            entry = {k: v for k, v in g.items() if k not in HIDDEN_GENE_FIELDS}

            # Validate required fields
            for field in ["essential", "has_human_homolog", "binding_compounds"]:
                if field not in g:
                    raise ValueError(f"Gene missing required field: {field}")

            entry["essential_score"] = self._noisy(g["essential"], 0.7, 0.25)
            entry["safety_score"] = self._noisy(not g["has_human_homolog"], 0.7, 0.25)
            entry["drug_likelihood"] = self._noisy(len(g["binding_compounds"]) > 0, 0.65, 0.3)

            result.append(entry)

        return result

    def _noisy(self, truth, hi, lo):
        center = hi if truth else lo
        return max(0.0, min(1.0, center + self._rng.gauss(0, PRIOR_NOISE)))

    # ---------------- LOAD ---------------- #

    def _load_databases(self):
        for key in ORGANISMS:
            path = os.path.join(DATA_DIR, f"genes_{key}.json")

            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing dataset file: {path}")

            with open(path) as f:
                data = json.load(f)

            if "genes" not in data:
                raise ValueError(f"Invalid dataset format in {path}")

            if len(data["genes"]) == 0:
                raise ValueError(f"Empty dataset: {path}")

            self._gene_db[key] = data