import requests
import os
from dotenv import load_dotenv
import yaml
import re

errorCounter = 0

# ------------------------------------------------------------------------------------------------------------------------------------ #

# PER CACHE
import sqlite3

def build_cache_key(prompt, model, temperature, seed):
    return f"{model}||{temperature}||{seed}||{prompt}"

# funzione per prendere dalla cache
def get_cache(key):
    cursor.execute("""
        SELECT response FROM llm_cache
        WHERE key = ?
    """, (key,))
    row = cursor.fetchone()
    return row[0] if row else None

# funzione per salvare nella cache una row
def save_cache(key, response):
    cursor.execute("""
        INSERT OR REPLACE INTO llm_cache (key, response)
        VALUES (?, ?)
    """, (key, response))
    conn.commit()

# ------------------------------------------------------------------------------------------------------------------------------------ #

# PYDANTIC PER GRAMMAR CONSTRAINED DECODER
from pydantic import BaseModel, create_model, constr
from typing import List, Tuple
import json

# modello per problem_name
def ProblemIDModel(types):
    field_type = map_type(types[0])

    return create_model(
        "ProblemIDModel",
        problem_name=(field_type, ...)
    )

# modello per predicate(nome_predicato)
def PredicateListModel():
    
    return create_model(
        "PredicateListModel",
        predicates=(list[str], ...)
    )

def map_type(t: str):
    if t == "int":
        return int
    elif t == "symbol":
        return constr(min_length=1, strip_whitespace=True)
    else:
        raise ValueError(f"Unknown type: {t}")


# creare il modello in base al predicato da chiedere alla llm
def create_predicate_model(predicate_name, types):

    if types is None:
        raise ValueError(f"Missing types for predicate: {predicate_name}")

    # CASE 1: boolean predicate (no args)
    if types == []:
        field_type = bool

    # CASE 2: unary predicate → List[T]
    elif len(types) == 1:
        field_type = List[map_type(types[0])]

    # CASE 3: n-ary predicate → List[Tuple[...]]
    else:
        py_types = tuple(map_type(t) for t in types)
        field_type = List[Tuple[py_types]]

    return create_model(
        f"{predicate_name}_Model",
        **{predicate_name: (field_type, ...)}
    )

# FUNZIONE PER TRASFORMARE L'OUTPUT IN FORMATO JSON DEI PREDICATI -> IN ATOMI DA USARE NELLA KB CON DLV/Clingo
def json_to_atoms(data, types):

    atoms = []

    for predicate, values in data.items():

        if types == []:
            if values:  # true
                atoms.append(f"{predicate}")
            continue

        # values is already: [(1,10,'s'), (1,11,'n')]
        if not isinstance(values, list):
            values = [values]

        for row in values:

            if not isinstance(row, (list, tuple)):
                row = [row]

            processed_args = []

            for i, v in enumerate(row):
                t = types[i]

                if t == "int":
                    processed_args.append(str(int(v)))
                elif t == "symbol":
                    v = str(v)
                    v = v.replace(" ", "_")
                    processed_args.append(v)

                else:
                    raise ValueError(f"Unsupported type: {t}")

            atoms.append(f"{predicate}({','.join(processed_args)})")

    return atoms

# FUNZIONE DI PARSING SU UN PREDICATO E RICAVARNE NOME E ARITÀ
def parse_predicate_signature(signature):

    name = signature.split("(")[0]

    inside = re.search(r"\((.*?)\)", signature)

    if inside:
        args = inside.group(1).split(",")
        arity = len(args)
    else:
        arity = 0

    return name, arity


# ------------------------------------------------------------------------------------------------------------------------------------ #

# PER CLINGO KB SOLVER
import clingo
import traceback

# FUNZIONE PER RICAVARE ULTERIORI FATTI DA UNA KB CON DLV/Clingo
def solve_with_asp(known_atoms, kb_content):
    try:
        """
        Solve ASP using Clingo.
        kb_content can be:
        - filename
        - inline ASP
        """

        ctl = clingo.Control(["--models=1"])
        program_parts = []

        is_inline = "\n" in kb_content or ":-" in kb_content or "#show" in kb_content

        if not is_inline:
            kb_path = f"./KNOWLEDGE_BASE_FILES/{kb_content}.dlv"
            if os.path.exists(kb_path):
                with open(kb_path, "r") as f:
                    program_parts.append(f.read())
        else:
            program_parts.append(kb_content)

        for atom in known_atoms:
            program_parts.append(atom.strip() + ".")

        program = "\n".join(program_parts)

        ctl.add("base", [], program)
        ctl.ground([("base", [])])

        output_atoms = set()

        def on_model(model):
            for sym in model.symbols(shown=True):
                output_atoms.add(str(sym))

        result = ctl.solve(on_model=on_model)

        global metrics
        global group
        metrics[group]["solver_calls"] += 1
        #print(f"[ASP SOLVER CALL] KB: {kb_content} | Known atoms: {len(known_atoms)} | Output atoms: {len(output_atoms)}")

        if not result.satisfiable:
            return known_atoms

        return output_atoms
    except Exception as e:

        global errorCounter
        errorCounter += 1

        print("[CLINGO EXCEPTION CAUGHT]")
        print("KB content:", kb_content)
        print("\n--- TRACEBACK ---")
        traceback.print_exc()
        print("="*80 + "\n")
        
        return known_atoms


# FUNZIONE PER RICAVARE ULTERIORI ATOMI DA UNA KB CON DLV/Clingo
def asp_condition_holds(if_condition, known_atoms, kb_path=None):
    try:
        ctl = clingo.Control(["--models=1"])

        program_parts = []

        for atom in known_atoms:
            program_parts.append(atom.strip() + ".")

        program_parts.append(f"ok :- {if_condition}.")
        program_parts.append(":- not ok.")

        program = "\n".join(program_parts)

        ctl.add("base", [], program)
        ctl.ground([("base", [])])

        result = ctl.solve()

        global metrics
        global group
        metrics[group]["solver_calls"] += 1
        #print(f"[ASP SOLVER CALL - CONDITION CHECK] Condition: {if_condition} | Known atoms: {len(known_atoms)} | Satisfiable: {result.satisfiable}")

        return result.satisfiable
    
    except Exception as e:

        global errorCounter
        errorCounter += 1

        print("[CLINGO EXCEPTION CAUGHT]")
        print("If-condition:", if_condition)
        print("\n--- TRACEBACK ---")
        traceback.print_exc()
        print("="*80 + "\n")



# ----------------------------------------------------------------------------- #

# LOGICA LETTURA FILE YAML

def recursive_predicate_explore(predicates, known_atoms, strings_context):
    retry_blocks = []
    explore_level(predicates, known_atoms, strings_context, retry_blocks)

    for pred in retry_blocks:
        explore_single_block(pred, known_atoms, strings_context)

def explore_level(predicates, known_atoms, strings_context, retry_blocks):

    for pred in predicates:

        name = list(pred.keys())[0]
        content = pred[name]

        # handle structural node FIRST
        if name == "_":
            pred_if = content.get("if")

            if pred_if:
                if not asp_condition_holds(pred_if, known_atoms):
                    if content.get("retry") == True:
                        retry_blocks.append(pred)
                    continue

            nested = content.get("predicates")
            if nested:
                explore_level(nested, known_atoms, strings_context, retry_blocks)

            continue  # IMPORTANT: never go to explore_single_block

        # normal predicate node logic
        pred_if = content.get("if")

        if pred_if:
            if not asp_condition_holds(pred_if, known_atoms):
                if content.get("retry") == True:
                    retry_blocks.append(pred)
                continue

        explore_single_block(pred, known_atoms, strings_context)

        nested = content.get("predicates")
        if nested:
            explore_level(nested, known_atoms, strings_context, retry_blocks)

def explore_single_block(pred, known_atoms, strings_context):

    new_atoms = []

    name = list(pred.keys())[0]

    if name == "_":
        return
    
    content = pred[name]

    pred_prompt = content.get("prompt")

    if pred_prompt:

        template = behavior_data["preprocessing"]["mapping"]
        template = template.replace("{input}", user_input)
        template = template.replace("{atom}", name)
        template = template.replace("{instructions}", pred_prompt.strip())

        for key, value in strings_context.items():
            template = template.replace(f"{{{key}}}", value)

        predicate_name, _ = parse_predicate_signature(name)

        if predicate_name == "predicate":
            Model = PredicateListModel()
            types = None
        else:   
            types = content.get("types")
            Model = create_predicate_model(predicate_name, types)

        # LLM CALL (same as before)
        key = build_cache_key(system_content+template, model_id, temperatureN, seedN)
        cached_response = get_cache(key)

        if cached_response:
            result = json.loads(cached_response)
            output = result["choices"][0]["message"]["content"]
            #print(f"[Cache found] Structured LLM output: {output}")
        else:
            schema = Model.model_json_schema()

            response = requests.post(
                generation_url,
                headers=headers,
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": template}
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": f"{predicate_name}_schema",
                            "schema": schema
                        }
                    },
                    "max_tokens": 5000,
                    "temperature": temperatureN,
                    "seed": seedN
                }
            )

            if response.status_code == 200:
                result = response.json()
                output = result["choices"][0]["message"]["content"]
                save_cache(key, json.dumps(result))

                #if predicate_name == "predicate":
                    #print(f"Identified predicates: {output}")
                #else:
                    #print(f"Structured LLM output for {name}:\n{output}\n")

            else:
                print(f"Error {response.status_code}: {response.text}")
                return

        data = json.loads(output)

        # print(output)

        validated = Model(**data)
        if predicate_name == "predicate":
            preds = validated.model_dump()["predicates"]
            new_atoms = [f"predicate({p})" for p in preds]
            
        else:
            new_atoms = json_to_atoms(validated.model_dump(), types)

    # --- KB ---
    kb_name = content.get("kb")

    if kb_name:
        # create temporary knowledge base
        temp_atoms = known_atoms.union(new_atoms)

        asp_atoms = solve_with_asp(temp_atoms, kb_name)

        # update known_atoms ONLY with ASP result
        known_atoms.update(asp_atoms)

    else:
        # if no kb, then normal behavior
        known_atoms.update(new_atoms)

# ----------------------------------------------------------------------------- #

# FILTRARE PREDICATI DI TIPO 'predicate()' ALLA FINE, NON ATTINENTI AL PROBLEMA

def filter_problem_atoms(atoms):
    filtered = set()

    for atom in atoms:
        name = atom.split("(")[0]

        if name == "predicate" or name == "problem_name":
            continue

        filtered.add(atom)

    return filtered

# ----------------------------------------------------------------------------- #

# PER MISURARE METRICHE DI PERFORMANCE

from collections import defaultdict

def normalize_atoms(atom_string_or_list, group=None):

    if isinstance(atom_string_or_list, str):
        raw = atom_string_or_list.replace("\n", " ").split(".")
    else:
        raw = atom_string_or_list

    atoms = set()

    for a in raw:
        a = a.strip()
        if not a:
            continue

        if a.endswith("."):
            a = a[:-1]

        # remove ALL spaces
        a = a.replace(" ", "")

        # G-SPECIFIC FIXES 
        if group == "GLNRS":

            # 1. remove type(...)
            if a.startswith("type("):
                continue

            # 2. warning spacing fix (extra safety)
            if a.startswith("warning("):
                a = a.replace(" ", "")

        atoms.add(a)

    return atoms

def compute_metrics(pred_atoms, gold_atoms):
    tp = len(pred_atoms & gold_atoms)
    fp = len(pred_atoms - gold_atoms)
    fn = len(gold_atoms - pred_atoms)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    perfect = 1.0 if pred_atoms == gold_atoms else 0.0

    return f1, perfect

def normalize_group(name):
    mapping = {
        "layered_graph": "GLNRS",
        "Labyrinth": "GLNRS",
        "Nomystery": "GLNRS",
        "Ricochet Robots": "GLNRS",
        "Sokoban": "GLNRS",
        1: "E",
        2: "E",
        3: "E"
    }
    return mapping.get(name, name)

group = None



metrics = defaultdict(lambda: {
        "f1_sum": 0,
        "perfect": 0,
        "count": 0,
        "solver_calls": 0
    })


# --------------------------------------------------------------------------------------------------------------------------  #
# --------------------------------------------------------------------------------------------------------------------------  #
# --------------------------------------------------------------------------------------------------------------------------  #
# --------------------------------------------------------------------------------------------------------------------------  #
# --------------------------------------------------------------------------------------------------------------------------  #
# --------------------------------------------------------------------------------------------------------------------------  #
# --------------------------------------------------------------------------------------------------------------------------  #
# --------------------------------------------------------------------------------------------------------------------------  #
# --------------------------------------------------------------------------------------------------------------------------  #

def main():

    # SETUP CONNESSIONE, VARIABILI GLOBALI, CARICAMENTO FILE YAML

    global conn, cursor, headers, behavior_data, app_data
    global system_content, model_id, generation_url
    global seedN, temperatureN, user_input

    load_dotenv()

    server_url = "https://laia-3.alviano.net:11444"
    api_key = os.getenv("OPENAI_API_KEY")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # CHIAMATA AL SERVER PER OTTENERE LA LISTA DEI MODELLI DISPONIBILI
    url = f"{server_url}/v1/models"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        models = response.json()
        models_list = [model['id'] for model in models['data']]
        print("Model IDs:", models_list)
    else:
        print(f"Error {response.status_code}: {response.text}")



    model_id = "llama3.1:8b"
    generation_url = f"{server_url}/v1/chat/completions"

    seedN = 5
    temperatureN = 0

    # TEST FATTI PER GLNRS
    # APPLICATION/Flat_G+LNRS.yml
    # APPLICATION/G+LNRS_Hierarchical_Test.yml

    # TEST PER G E LNRS SEPARATI
    # APPLICATION/Flat_G.yml
    # APPLICATION/LNRS_Hierarchical.yml
    # APPLICATION/Flat_LNRS.yml
    
    # TEST PER E
    # APPLICATION/Flat_ElectricalGrid.yml
    # APPLICATION/Hierarchy_ElectricalGrid2.yml

    # BEHAVIOR/behaviour_HGX.yml

    """
    RICORDA DI CAMBIARE NEL CODICE SE TESTI SU G O GLNRS, MODIFICA IL DICTIONARY PER I RISULTATI, E MODIFICA L'IF NEL MAIN PER
    ESCLUDERE POTENZIALMENTE I GRUPPI NON UTILI.
    """
    
    # DATASET/dataset_G+LNRS.json
    # DATASET/dataset_E.json

    with open("BEHAVIOR/behaviour_HGX.yml", "r", encoding="utf-8") as f:
        behavior_data = yaml.safe_load(f)

    with open("APPLICATION/Hierarchy_ElectricalGrid2.yml", "r") as f:
        app_data = yaml.safe_load(f)

    with open("DATASET/dataset_E.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    use_extract_mode = "extract" in app_data

    conn = sqlite3.connect("llm_cache.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS llm_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT UNIQUE,
        response TEXT
    )
    """)
    conn.commit()

    system_content_template = (
        behavior_data["preprocessing"]["init"] + "\n" +
        behavior_data["preprocessing"]["context"]
    )

    system_content = system_content_template.replace(
        "{context}",
        app_data["preprocessing"][0]["_"]
    )


    
    
    for i, sample in enumerate(dataset):

        """
        group = normalize_group(sample["problem_name"])
        if group != "G":
            continue
        """

        print("\n" + "="*80)
        print(f"ITEM {i} | {sample['problem_name']}")
        print("="*80)


        global group
        group = normalize_group(sample["problem_name"])

        if group != "E":
            user_input = sample["text"]
        
            # ---------------- PROBLEM NAME ----------------

            if use_extract_mode:

                has_problem_name = (
                    use_extract_mode and
                    len(app_data["extract"]) > 0 and
                    "problem_name(problem)" in app_data["extract"][0]
                )
                if has_problem_name:
                    problem_block = app_data["extract"][0]["problem_name(problem)"]

                else:
                    problem_block = None

            else:
                problem_block = app_data["preprocessing"][1]["problem_name(problem)"]

            problem_entry = None
            if use_extract_mode and not has_problem_name:
                problem_entry = {
                    "predicates": app_data["extract"]
                }

            if problem_block is not None:
                instructions = problem_block["prompt"]

                source = app_data["extract"] if use_extract_mode else app_data["preprocessing"]
                first_entry = source[0]
                atom = list(first_entry.keys())[0]

                types = problem_block["types"]

                template = behavior_data["preprocessing"]["mapping"]
                template = template.replace("{input}", user_input)
                template = template.replace("{atom}", atom)
                template = template.replace("{instructions}", instructions.strip())

                key = build_cache_key(system_content+template, model_id, temperatureN, seedN)
                cached_response = get_cache(key)

                if cached_response:
                    result = json.loads(cached_response)
                    problem_name = result["choices"][0]["message"]["content"]
                    #print(f"[Cache found] Structured LLM output: {problem_name}")
                else:
                    Model = ProblemIDModel(types)

                    response = requests.post(
                        generation_url,
                        headers=headers,
                        json={
                            "model": model_id,
                            "messages": [
                                {"role": "system", "content": system_content},
                                {"role": "user", "content": template}
                            ],
                            "response_format": {
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "problem_name_schema",
                                    "schema": Model.model_json_schema()
                                }
                            },
                            "max_tokens": 500,
                            "temperature": temperatureN,
                            "seed": seedN
                        }
                    )

                    if response.status_code == 200:
                        result = response.json()
                        problem_name = result["choices"][0]["message"]["content"]

                        save_cache(key, json.dumps(result))

                        print(f"Identified problem: {problem_name}")
                    else:
                        print(f"Error phase 1 (SERVER ERROR): {response.status_code}: {response.text}")
                        problem_name = 1

                # ---------------- FIND PROBLEM ----------------

                problem_name_typ = json.loads(problem_name)["problem_name"]
                problem_atom = f"problem_name({problem_name_typ})"

                if not use_extract_mode:
                    for entry in app_data["preprocessing"][1]["problem_name(problem)"]["predicates"]:
                        block = entry.get("_")
                        if block and block.get("if", "").strip() == problem_atom:
                            problem_entry = block
                            break
                else:
                    # FLAT SYSTEM
                    problem_entry = {
                            "predicates": app_data["extract"][1:]  # skippo problem_name, già fatto
                    }
                    

                if problem_entry is None:
                    raise ValueError(f"No matching problem block found for problem_atom: {problem_atom}")

            # ---------------- STRINGS ----------------

            strings_context = {}

            # FLAT MODE
            if use_extract_mode:
                for entry in app_data.get("strings", []):
                    for k, v in entry.items():
                        strings_context[k] = v
            else:
                strings_context = {}
                for entry in (problem_entry.get("strings") or []):
                    for k, v in entry.items():
                        strings_context[k] = v

            # ---------------- FIND PREDICATES ----------------

            known_atoms = set()
            if problem_entry is not None and problem_block is not None:
                known_atoms.add(problem_atom) 
  
            # ---------------- HIERARCHICAL / FLAT SOLVING ----------------

            if not use_extract_mode:
                recursive_predicate_explore(
                    problem_entry.get("predicates", []),
                    known_atoms,
                    strings_context
                )
            else:
                retry_blocks = []

                for pred in problem_entry["predicates"]:
                    name = list(pred.keys())[0]
                    content = pred[name]

                    pred_if = content.get("if")

                    if pred_if and not asp_condition_holds(pred_if, known_atoms):
                        if content.get("retry") == True:
                            retry_blocks.append(pred)
                        continue

                    explore_single_block(pred, known_atoms, strings_context)
                for pred in retry_blocks:
                    explore_single_block(pred, known_atoms, strings_context)

            # ---------------- FINAL FILTER ----------------

            final_facts = filter_problem_atoms(known_atoms)


            #print(final_facts)

            pred_atoms = normalize_atoms(final_facts, group)
            gold_atoms = normalize_atoms(sample["output"], group)
            

            

            print("------------------------------------------------------------------------")
            print(pred_atoms)
            print("------------------------------------------------------------------------")
            print(gold_atoms)
            print("------------------------------------------------------------------------")
            if (pred_atoms) != (gold_atoms):
                if len(pred_atoms) > len(gold_atoms):
                    print("EXTRA PREDICTED ATOMS:")
                    print(pred_atoms - gold_atoms)
                if len(gold_atoms) > len(pred_atoms):
                    print("MISSED GOLD ATOMS:")
                    print(gold_atoms - pred_atoms)
                if len(pred_atoms) == len(gold_atoms):
                    print("SAME NUMBER OF ATOMS, CHECK DIFFERENCES:")
                    print("PREDICTED NOT IN GOLD:")
                    print(pred_atoms - gold_atoms)
        
            

            f1, perfect = compute_metrics(pred_atoms, gold_atoms)

            global metrics

            metrics[group]["f1_sum"] += f1
            metrics[group]["perfect"] += perfect
            metrics[group]["count"] += 1

        
        
        
    
    for group, m in metrics.items():
        count = m["count"]

        avg_f1 = m["f1_sum"] / count if count > 0 else 0
        perfect_rate = m["perfect"] / count if count > 0 else 0
        solver_calls = m["solver_calls"]

        print("\n========================")
        print("GROUP:", group)
        print("F1:", avg_f1)
        print("Perfect rate:", perfect_rate)
        print("Solver calls:", solver_calls)
        print("Count:", count)


    if errorCounter > 0:
        print("\nIncorrect Clingo syntax events:", errorCounter)
    

if __name__ == "__main__":
    main()