import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
# Ensure necessary libraries are imported
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
except ImportError:
    print("Please install transformers and torch: pip install transformers torch accelerate")
    exit()

class ClinIQLinkSampleDatasetSubmit:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.qa_dir = os.path.join(self.base_dir, "..", "sample_QA_pairs")
        self.template_dir = os.path.join(self.base_dir, "submission_template")
        # Load a pre-trained SentenceTransformer model for semantic similarity calculations.
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            print("NLTK 'punkt' resource not found. Downloading...", flush=True)
            nltk.download('punkt')
        except Exception as e:
             print(f"Error checking/downloading NLTK 'punkt': {e}", flush=True)
             # Decide if you want to exit or continue without punkt potentially affecting sentence tokenization
             # exit()

        # --- Model and Pipeline Initialization ---
        # Attributes to store models, tokenizers, and pipelines
        self.generator_model = None
        self.generator_tokenizer = None
        self.verifier_model = None
        self.verifier_tokenizer = None
        self.generator_pipeline = None
        self.verifier_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}", flush=True)

        # Load models and pipelines immediately
        self.load_participant_model()
        self.load_participant_pipeline()
        # --- End Model and Pipeline Initialization ---


    # --- MODIFIED FUNCTION 1: Load Models and Tokenizers ---
    def load_participant_model(self):
        """
        Loads the chosen LLM models (Generator: Gemma3-27B, Verifier: QWQ-32B)
        and their tokenizers locally.
        Stores them as instance attributes.
        """
        print("Loading participant LLM models and tokenizers...", flush=True)
        # --- Load Generator Model (Gemma3-27B Placeholder) ---
        generator_model_id = "google/gemma3-27b-placeholder" # <<< REPLACE WITH ACTUAL IDENTIFIER
        print(f"Attempting to load Generator: {generator_model_id}", flush=True)
        try:
            self.generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_id)
            # Consider device_map="auto" if using accelerate for large models
            self.generator_model = AutoModelForCausalLM.from_pretrained(
                generator_model_id,
                torch_dtype=torch.bfloat16, # Or float16, adjust based on model/hardware
                device_map=self.device # Use "auto" if using multiple GPUs or offloading
            )
            # Ensure pad token is set for tokenizer
            if self.generator_tokenizer.pad_token is None:
                self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token
                self.generator_model.config.pad_token_id = self.generator_model.config.eos_token_id

            print(f"Generator model '{generator_model_id}' loaded successfully.", flush=True)
        except Exception as e:
            print(f"CRITICAL ERROR loading Generator model '{generator_model_id}': {e}", flush=True)
            print("Ensure the model identifier is correct and dependencies are installed.", flush=True)
            # Decide on error handling: maybe raise error or set model to None
            self.generator_model = None
            self.generator_tokenizer = None
            # raise e # Or handle more gracefully depending on competition rules

        # --- Load Verifier Model (QWQ-32B Placeholder) ---
        verifier_model_id = "qwq/qwq-32b-placeholder" # <<< REPLACE WITH ACTUAL IDENTIFIER
        print(f"Attempting to load Verifier: {verifier_model_id}", flush=True)
        try:
            self.verifier_tokenizer = AutoTokenizer.from_pretrained(verifier_model_id)
            self.verifier_model = AutoModelForCausalLM.from_pretrained(
                verifier_model_id,
                torch_dtype=torch.bfloat16, # Or float16
                device_map=self.device # Use "auto" if needed
            )
             # Ensure pad token is set for tokenizer
            if self.verifier_tokenizer.pad_token is None:
                self.verifier_tokenizer.pad_token = self.verifier_tokenizer.eos_token
                self.verifier_model.config.pad_token_id = self.verifier_model.config.eos_token_id

            print(f"Verifier model '{verifier_model_id}' loaded successfully.", flush=True)
        except Exception as e:
            print(f"CRITICAL ERROR loading Verifier model '{verifier_model_id}': {e}", flush=True)
            print("Ensure the model identifier is correct and dependencies are installed.", flush=True)
            self.verifier_model = None
            self.verifier_tokenizer = None
            # raise e # Or handle gracefully

        # Original function didn't return anything useful, so we just set attributes.

    # --- MODIFIED FUNCTION 2: Load Inference Pipelines ---
    def load_participant_pipeline(self):
        """
        Initializes the LLM inference pipelines for both loaded models.
        Stores them as instance attributes.
        """
        print("Loading participant LLM pipelines...", flush=True)

        # --- Initialize Generator Pipeline ---
        if self.generator_model and self.generator_tokenizer:
            print(f"Initializing Generator pipeline for '{self.generator_model.config._name_or_path}'", flush=True)
            try:
                self.generator_pipeline = pipeline(
                    "text-generation",
                    model=self.generator_model,
                    tokenizer=self.generator_tokenizer,
                    device=self.device if self.device=='cpu' else 0 # pipeline device maps differently
                )
                print("Generator pipeline loaded successfully.", flush=True)
            except Exception as e:
                print(f"Error initializing Generator pipeline: {e}", flush=True)
                self.generator_pipeline = None
        else:
            print("Generator model/tokenizer not loaded, cannot initialize pipeline.", flush=True)
            self.generator_pipeline = None

        # --- Initialize Verifier Pipeline ---
        if self.verifier_model and self.verifier_tokenizer:
            print(f"Initializing Verifier pipeline for '{self.verifier_model.config._name_or_path}'", flush=True)
            try:
                self.verifier_pipeline = pipeline(
                    "text-generation",
                    model=self.verifier_model,
                    tokenizer=self.verifier_tokenizer,
                    device=self.device if self.device=='cpu' else 0 # pipeline device maps differently
                )
                print("Verifier pipeline loaded successfully.", flush=True)
            except Exception as e:
                print(f"Error initializing Verifier pipeline: {e}", flush=True)
                self.verifier_pipeline = None
        else:
            print("Verifier model/tokenizer not loaded, cannot initialize pipeline.", flush=True)
            self.verifier_pipeline = None

        # Original function didn't return anything useful.

    # --- MODIFIED FUNCTION 3: Call Appropriate Model and Get Response ---
    def generate_or_verify_response(self, prompt):
        """
        Determines which model (Generator or Verifier) to use based on the prompt content,
        calls the appropriate pipeline, and returns the generated text.

        Args:
            prompt (str): The formatted prompt string for the LLM.

        Returns:
            str: The processed response text from the chosen LLM.
        """
        # Determine which pipeline to use based on prompt keywords
        # These keywords are based on the prompt templates provided in the sample repo
        use_verifier = False
        if "Incorrect Reasoning Step:" in prompt or "identify the specific reasoning step" in prompt:
            print("[INFO] Using Verifier Model (QWQ-32B) for Multi-Hop Inverse Task.", flush=True)
            use_verifier = True
            active_pipeline = self.verifier_pipeline
            model_name = "Verifier (QWQ-32B Placeholder)"
        elif "Incorrect Explanation:" in prompt or "explanation of why it is wrong" in prompt:
            print("[INFO] Using Verifier Model (QWQ-32B) for Short Inverse Task.", flush=True)
            use_verifier = True
            active_pipeline = self.verifier_pipeline
            model_name = "Verifier (QWQ-32B Placeholder)"
        else:
            # Default to Generator for TF, MC, List, Short, standard Multi-Hop
            print("[INFO] Using Generator Model (Gemma3-27B) for Generation Task.", flush=True)
            active_pipeline = self.generator_pipeline
            model_name = "Generator (Gemma3-27B Placeholder)"

        # Check if the required pipeline is available
        if active_pipeline is None:
            error_msg = f"ERROR: The required {model_name} pipeline is not available. Check loading errors."
            print(error_msg, flush=True)
            return error_msg # Return an error message or handle appropriately

        # --- Generate Response ---
        try:
            # Set generation parameters (adjust as needed)
            # max_new_tokens is generally preferred over max_length for pipelines
            # to control *added* text length.
            max_new_toks = 512 # Adjust based on expected answer length for different tasks
            if use_verifier:
                max_new_toks = 256 # Maybe shorter responses needed for verification tasks
            elif "Reasoning: <your step-by-step reasoning here>" in prompt: # Multi-hop
                max_new_toks = 768 # Allow more tokens for reasoning

            print(f"[DEBUG] Calling {model_name} with max_new_tokens={max_new_toks}", flush=True)
            # The pipeline returns a list of dictionaries
            outputs = active_pipeline(
                prompt,
                max_new_tokens=max_new_toks,
                num_return_sequences=1,
                do_sample=False, # Use deterministic output for consistency? Or True for more varied?
                # top_k=50,       # Example sampling params if do_sample=True
                # top_p=0.95,     # Example sampling params if do_sample=True
                # temperature=0.7,# Example sampling params if do_sample=True
                pad_token_id=active_pipeline.tokenizer.eos_token_id # Suppress warning
            )

            # Extract the generated text
            raw_generated_text = outputs[0]['generated_text']

            # --- Post-process: Remove the input prompt from the generated text ---
            # Check if the generated text starts with the prompt
            if raw_generated_text.startswith(prompt):
                processed_text = raw_generated_text[len(prompt):].strip()
            else:
                # Fallback if the model didn't perfectly echo the prompt (less common with newer models)
                # This might need refinement depending on model behavior
                print("[WARN] Model output didn't start with the exact prompt. Returning full output.", flush=True)
                processed_text = raw_generated_text.strip() # Return everything for inspection

            print(f"[DEBUG] {model_name} Raw Output Length: {len(raw_generated_text)}", flush=True)
            print(f"[DEBUG] {model_name} Processed Output Length: {len(processed_text)}", flush=True)
            # print(f"[DEBUG] Processed Output Sample: {processed_text[:100]}...", flush=True) # Optional debug print

            return processed_text

        except Exception as e:
            error_msg = f"ERROR during {model_name} inference: {e}"
            print(error_msg, flush=True)
            # Consider how critical this is. Maybe return error string or raise?
            return error_msg # Indicate failure


    # --- Helper function to load JSON (Unmodified) ---
    def load_json(self, filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON from {filepath}: {e}", flush=True)
            return None

    # --- Helper function to load Template (Unmodified) ---
    def load_template(self, filename):
        filepath = os.path.join(self.template_dir, filename)
        try:
            with open(filepath, "r") as f:
                return f.read()
        except Exception as e:
            print(f"Error loading template {filename} from {filepath}: {e}", flush=True)
            return None

    # --- Prompt Generation Logic (Unmodified) ---
    def generate_prompt(self, template, qa, qa_type):
        try:
            question = qa.get("question", "Unknown Question")
            answer = qa.get("answer", "") # Used in inverse prompts
            options = qa.get("options", {})
            reasoning = qa.get("reasoning", "") # Used in inverse prompts
            false_answer = qa.get("false_answer", "") # Used in inverse prompts
            incorrect_reasoning_step = qa.get("incorrect_reasoning_step", "") # MH Inverse Ground Truth (not used in prompt)
            incorrect_explanation = qa.get("incorrect_explanation", "") # Short Inverse Ground Truth (not used in prompt)

            if qa_type == "true_false":
                return template.format(question=question)
            elif qa_type == "multiple_choice":
                 # Ensure the options placeholders match the MC template exactly
                options_map = {
                    "A": options.get("A", "Option A missing"),
                    "B": options.get("B", "Option B missing"),
                    "C": options.get("C", "Option C missing"),
                    "D": options.get("D", "Option D missing")
                 }
                return template.format(question=question, **options_map)
            elif qa_type == "list":
                # Format list options clearly for the prompt
                options_list = qa.get("options", [])
                options_formatted = "\n".join([f"- {opt}" for opt in options_list]) if isinstance(options_list, list) else str(options)
                return template.format(question=question, options_joined=options_formatted)
            elif qa_type == "multi_hop":
                # Standard multi-hop asks for answer + reasoning
                 return template.format(question=question)
            elif qa_type == "multi_hop_inverse":
                # Inverse multi-hop provides question, answer, reasoning -> asks to identify flawed step
                 # Ensure reasoning is formatted reasonably for the prompt if it's a list
                reasoning_str = "\n".join(reasoning) if isinstance(reasoning, list) else str(reasoning)
                return template.format(question=question, answer=answer, reasoning=reasoning_str)
            elif qa_type == "short":
                return template.format(question=question)
            elif qa_type == "short_inverse":
                 # Inverse short asks to explain why the 'false_answer' is wrong
                return template.format(question=question, false_answer=false_answer)
            else:
                print(f"Warning: Unknown QA type '{qa_type}'", flush=True)
                return "Invalid QA type."
        except Exception as e:
            print(f"Error generating prompt: {e}", flush=True)
            return "Error generating prompt."

    # --- Evaluation Logic (Unmodified - Keep all these evaluation functions as they are) ---
    def compute_f1_score(self, true_list, pred_list):
        try:
            true_set = set(item.strip().lower() for item in true_list)
            pred_set = set(item.strip().lower() for item in pred_list)
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            return precision, recall, f1
        except Exception as e:
            print(f"Error computing F1 score: {e}", flush=True)
            return 0.0, 0.0, 0.0

    def evaluate_true_false(self, expected, prediction):
        try:
            # Be slightly more robust to variations like "True." vs "True"
            return 1.0 if expected.strip().lower().rstrip('.') == prediction.strip().lower().rstrip('.') else 0.0
        except Exception as e:
            print(f"Error evaluating True/False question: {e}", flush=True)
            return 0.0

    def evaluate_multiple_choice(self, expected, prediction):
        try:
             # Prediction should be a single letter A, B, C, or D
            return 1.0 if expected.strip().upper() == prediction.strip().upper() else 0.0
        except Exception as e:
            print(f"Error evaluating Multiple Choice question: {e}", flush=True)
            return 0.0

    def evaluate_list(self, expected, prediction):
        try:
            # Prediction should be a comma-separated list of letters/items
            # Normalize prediction: Convert letters to items or vice-versa if needed, handle spaces
            # Assuming prediction is a comma-separated string here based on template
            pred_list = [item.strip().lower() for item in prediction.split(",") if item.strip()]
            exp_list = [item.strip().lower() for item in expected]
            _, _, f1 = self.compute_f1_score(exp_list, pred_list)
            return f1
        except Exception as e:
            print(f"Error evaluating List question: {e}", flush=True)
            return 0.0

    def compute_word_level_similarity(self, expected_text, prediction_text):
        try:
            expected_words = expected_text.split()
            prediction_words = prediction_text.split()
            if not expected_words or not prediction_words: return 0.0
            expected_embeds = self.st_model.encode(expected_words)
            prediction_embeds = self.st_model.encode(prediction_words)
            sim_matrix = cosine_similarity(expected_embeds, prediction_embeds)
            recall = np.mean(np.max(sim_matrix, axis=1))
            precision = np.mean(np.max(sim_matrix, axis=0))
            if (precision + recall) == 0: return 0.0
            return 2 * precision * recall / (precision + recall)
        except Exception as e:
            print(f"Error computing word-level similarity: {e}", flush=True)
            return 0.0

    def compute_sentence_level_similarity(self, expected_text, prediction_text):
        try:
            expected_sentences = nltk.sent_tokenize(expected_text)
            prediction_sentences = nltk.sent_tokenize(prediction_text)
            if not expected_sentences or not prediction_sentences: return 0.0
            expected_embeds = self.st_model.encode(expected_sentences)
            prediction_embeds = self.st_model.encode(prediction_sentences)
            sim_matrix = cosine_similarity(expected_embeds, prediction_embeds)
            # Average of max similarity for each expected sentence
            recall_focused_sim = np.mean(np.max(sim_matrix, axis=1))
            return recall_focused_sim
        except Exception as e:
            print(f"Error computing sentence-level similarity: {e}", flush=True)
            return 0.0

    def compute_paragraph_level_similarity(self, expected_text, prediction_text):
        try:
            expected_embed = self.st_model.encode([expected_text])
            prediction_embed = self.st_model.encode([prediction_text])
            sim = cosine_similarity(expected_embed, prediction_embed)[0][0]
            # Clip similarity to be between 0 and 1
            return max(0.0, min(1.0, sim))
        except Exception as e:
            print(f"Error computing paragraph-level similarity: {e}", flush=True)
            return 0.0

    def evaluate_open_ended(self, expected, prediction):
        try:
            # Normalize texts slightly before exact match check
            norm_expected = ' '.join(expected.strip().lower().split())
            norm_prediction = ' '.join(prediction.strip().lower().split())
            if norm_expected == norm_prediction:
                return 1.0
            if not norm_expected or not norm_prediction: # Handle empty strings
                 return 0.0

            word_sim = self.compute_word_level_similarity(norm_expected, norm_prediction)
            sentence_sim = self.compute_sentence_level_similarity(norm_expected, norm_prediction)
            paragraph_sim = self.compute_paragraph_level_similarity(norm_expected, norm_prediction)

            w_word, w_sentence, w_paragraph = 0.3, 0.3, 0.4
            semantic_score = w_word * word_sim + w_sentence * sentence_sim + w_paragraph * paragraph_sim
            semantic_score = max(0.0, min(1.0, semantic_score)) # Ensure score is [0, 1]

            # Linear scaling between thresholds
            low_threshold, high_threshold = 0.4, 0.9
            if semantic_score >= high_threshold: return 1.0
            if semantic_score < low_threshold: return 0.0
            return (semantic_score - low_threshold) / (high_threshold - low_threshold)

        except Exception as e:
            print(f"Error evaluating open-ended question: {e}", flush=True)
            return 0.0

    def evaluate_open_ended_metrics(self, expected, prediction):
        try:
            # Ensure inputs are strings and handle potential empty predictions
            expected_str = str(expected)
            prediction_str = str(prediction) if prediction else "" # Use empty string if prediction is None or empty
            expected_tokens = expected_str.split()
            prediction_tokens = prediction_str.split()

             # Avoid errors with empty predictions for BLEU/METEOR
            if not prediction_tokens:
                 return {"bleu": 0.0, "meteor": 0.0, "rouge": 0.0}
            if not expected_tokens: # Handle case where expected is empty? Score should be 0.
                 return {"bleu": 0.0, "meteor": 0.0, "rouge": 0.0}


            smoothing_function = SmoothingFunction().method1 # Common smoothing
            bleu = sentence_bleu([expected_tokens], prediction_tokens, smoothing_function=smoothing_function)

            # NLTK METEOR score needs wordnet data, ensure downloaded
            try:
                 nltk.data.find('corpora/wordnet')
            except nltk.downloader.DownloadError:
                 print("NLTK 'wordnet' resource not found for METEOR. Downloading...", flush=True)
                 nltk.download('wordnet')
            except Exception as e:
                  print(f"Error checking/downloading NLTK 'wordnet': {e}. METEOR score might fail.", flush=True)

            try:
                 # METEOR expects tokenized lists within a list for reference
                 meteor = meteor_score([expected_tokens], prediction_tokens)
            except LookupError:
                 print("[WARN] WordNet lookup failed for METEOR score, returning 0.0.", flush=True)
                 meteor = 0.0
            except Exception as e_meteor:
                 print(f"Error computing METEOR score: {e_meteor}", flush=True)
                 meteor = 0.0


            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            # ROUGE score handles empty strings gracefully
            rouge_scores = scorer.score(expected_str, prediction_str)
            rouge_avg = (rouge_scores['rouge1'].fmeasure + rouge_scores['rougeL'].fmeasure) / 2.0

            return {"bleu": bleu, "meteor": meteor, "rouge": rouge_avg}
        except Exception as e:
            print(f"Error evaluating open-ended metrics: {e}", flush=True)
            return {"bleu": 0.0, "meteor": 0.0, "rouge": 0.0}

    # --- Main Evaluation Loop Functions (Unmodified - these call the modified generate_or_verify_response) ---
    # Note: YOUR_LLM_PLACEHOLDER is replaced by generate_or_verify_response
    def evaluate_true_false_questions(self):
        try:
            tf_path = os.path.join(self.qa_dir, "TF.json")
            tf_data = self.load_json(tf_path)
            if tf_data is None: return {"average": 0.0, "scores": {}}
            template = self.load_template("tf_template.prompt")
            results = {}
            scores = []
            for qa in tf_data:
                try:
                    prompt = self.generate_prompt(template, qa, "true_false")
                    response = self.generate_or_verify_response(prompt) # Use the new function
                    expected = qa.get("answer", "")
                    score = self.evaluate_true_false(expected, response)
                    para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {"question": qa.get("question", ""), "expected": expected, "predicted": response, "score": score}
                    scores.append(score)
                except Exception as e: print(f"Error processing TF QA {qa.get('source', {}).get('paragraph_id', 'unknown')}: {e}", flush=True)
            overall_score = sum(scores) / len(scores) if scores else 0.0
            print(f"Overall True/False Score: {overall_score:.4f}", flush=True)
            return {"average": overall_score, "scores": results}
        except Exception as e: print(f"Error evaluating True/False questions: {e}", flush=True); return {"average": 0.0, "scores": {}}

    def evaluate_multiple_choice_questions(self):
        try:
            mc_path = os.path.join(self.qa_dir, "MC.json")
            mc_data = self.load_json(mc_path)
            if mc_data is None: return {"average": 0.0, "scores": {}}
            template = self.load_template("MC_template.prompt")
            results = {}
            scores = []
            for qa in mc_data:
                try:
                    prompt = self.generate_prompt(template, qa, "multiple_choice")
                    response = self.generate_or_verify_response(prompt) # Use the new function
                    expected = qa.get("correct_answer", "")
                    score = self.evaluate_multiple_choice(expected, response)
                    para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {"question": qa.get("question", ""), "expected": expected, "predicted": response, "score": score}
                    scores.append(score)
                except Exception as e: print(f"Error processing MC QA {qa.get('source', {}).get('paragraph_id', 'unknown')}: {e}", flush=True)
            overall_score = sum(scores) / len(scores) if scores else 0.0
            print(f"Overall Multiple Choice Score: {overall_score:.4f}", flush=True)
            return {"average": overall_score, "scores": results}
        except Exception as e: print(f"Error evaluating Multiple Choice questions: {e}", flush=True); return {"average": 0.0, "scores": {}}

    def evaluate_list_questions(self):
        try:
            list_path = os.path.join(self.qa_dir, "list.json")
            list_data = self.load_json(list_path)
            if list_data is None: return {"average": 0.0, "scores": {}}
            template = self.load_template("list_template.prompt")
            results = {}
            scores = []
            for qa in list_data:
                try:
                    prompt = self.generate_prompt(template, qa, "list")
                    response = self.generate_or_verify_response(prompt) # Use the new function
                    expected_items = qa.get("answer", [])
                    # Assuming response is comma-separated as per template comment
                    score = self.evaluate_list(expected_items, response)
                    para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {"question": qa.get("question", ""), "expected": expected_items, "predicted": response, "score": score}
                    scores.append(score)
                except Exception as e: print(f"Error processing List QA {qa.get('source', {}).get('paragraph_id', 'unknown')}: {e}", flush=True)
            overall_f1 = sum(scores) / len(scores) if scores else 0.0
            print(f"Overall List Question F1 Score: {overall_f1:.4f}", flush=True)
            return {"average": overall_f1, "scores": results}
        except Exception as e: print(f"Error evaluating List questions: {e}", flush=True); return {"average": 0.0, "scores": {}}

    def evaluate_short_questions(self):
        try:
            short_path = os.path.join(self.qa_dir, "short.json")
            short_data = self.load_json(short_path)
            if short_data is None: return {"average": 0.0, "scores": {}}
            template = self.load_template("short_template.prompt")
            results = {}
            scores = []
            for qa in short_data:
                try:
                    prompt = self.generate_prompt(template, qa, "short")
                    response = self.generate_or_verify_response(prompt) # Use the new function
                    expected = qa.get("answer", "")
                    f1_score = self.evaluate_open_ended(expected, response)
                    metrics = self.evaluate_open_ended_metrics(expected, response)
                    para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {"question": qa.get("question", ""), "expected": expected, "predicted": response, "f1_score": f1_score, "metrics": metrics}
                    scores.append(f1_score)
                except Exception as e: print(f"Error processing Short QA {qa.get('source', {}).get('paragraph_id', 'unknown')}: {e}", flush=True)
            avg_score = sum(scores) / len(scores) if scores else 0.0
            print(f"Average Short Answer Score (Semantic): {avg_score:.4f}", flush=True)
            return {"average": avg_score, "scores": results}
        except Exception as e: print(f"Error evaluating Short Answer questions: {e}", flush=True); return {"average": 0.0, "scores": {}}

    def evaluate_short_inverse_questions(self):
        try:
            short_inverse_path = os.path.join(self.qa_dir, "short_inverse.json")
            short_inverse_data = self.load_json(short_inverse_path)
            if short_inverse_data is None: return {"average": 0.0, "scores": {}}
            template = self.load_template("short_inverse_template.prompt")
            results = {}
            scores = []
            for qa in short_inverse_data:
                try:
                    prompt = self.generate_prompt(template, qa, "short_inverse")
                    response = self.generate_or_verify_response(prompt) # Use the new function
                    # Expected answer for evaluation is the ground truth 'incorrect_explanation'
                    expected = qa.get("incorrect_explanation", "")
                    f1_score = self.evaluate_open_ended(expected, response)
                    metrics = self.evaluate_open_ended_metrics(expected, response)
                    para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {"question": qa.get("question", ""), "expected": expected, "predicted": response, "f1_score": f1_score, "metrics": metrics}
                    scores.append(f1_score)
                except Exception as e: print(f"Error processing Short Inverse QA {qa.get('source', {}).get('paragraph_id', 'unknown')}: {e}", flush=True)
            avg_score = sum(scores) / len(scores) if scores else 0.0
            print(f"Average Short Inverse Score (Semantic): {avg_score:.4f}", flush=True)
            return {"average": avg_score, "scores": results}
        except Exception as e: print(f"Error evaluating Short Inverse questions: {e}", flush=True); return {"average": 0.0, "scores": {}}

    def evaluate_multi_hop_questions(self):
        try:
            mh_path = os.path.join(self.qa_dir, "multi_hop.json")
            mh_data = self.load_json(mh_path)
            if mh_data is None: return {"average": 0.0, "scores": {}}
            template = self.load_template("multi_hop_template.prompt")
            results = {}
            scores = []
            for qa in mh_data:
                try:
                    prompt = self.generate_prompt(template, qa, "multi_hop")
                    response = self.generate_or_verify_response(prompt) # Use the new function
                    # Expected answer for evaluation is the 'answer' field
                    expected = qa.get("answer", "")
                    # Need to parse response to separate 'Final Answer:' and 'Reasoning:' if template enforces it
                    # For simplicity now, evaluate semantic similarity on the whole response vs expected answer string.
                    # A more advanced evaluation might compare reasoning steps if provided.
                    # Let's assume the LLM mainly returns the answer first.
                    predicted_answer = response # Simplification: Treat whole response as answer for semantic score
                    # Extract answer if format is strict "Final Answer: ... Reasoning: ..."
                    if "Final Answer:" in response:
                         parts = response.split("Reasoning:", 1)
                         answer_part = parts[0].replace("Final Answer:", "").strip()
                         predicted_answer = answer_part

                    f1_score = self.evaluate_open_ended(expected, predicted_answer)
                    metrics = self.evaluate_open_ended_metrics(expected, predicted_answer)
                    para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {"question": qa.get("question", ""), "expected": expected, "predicted": response, "f1_score": f1_score, "metrics": metrics}
                    scores.append(f1_score)
                except Exception as e: print(f"Error processing Multi-hop QA {qa.get('source', {}).get('paragraph_id', 'unknown')}: {e}", flush=True)
            avg_score = sum(scores) / len(scores) if scores else 0.0
            print(f"Average Multi-hop Score (Semantic): {avg_score:.4f}", flush=True)
            return {"average": avg_score, "scores": results}
        except Exception as e: print(f"Error evaluating Multi-hop questions: {e}", flush=True); return {"average": 0.0, "scores": {}}

    def evaluate_multi_hop_inverse_questions(self):
        try:
            mh_inverse_path = os.path.join(self.qa_dir, "multi_hop_inverse.json")
            mh_inverse_data = self.load_json(mh_inverse_path)
            if mh_inverse_data is None: return {"average": 0.0, "scores": {}}
            template = self.load_template("multi_hop_inverse_template.prompt")
            results = {}
            scores = []
            for qa in mh_inverse_data:
                try:
                    prompt = self.generate_prompt(template, qa, "multi_hop_inverse")
                    response = self.generate_or_verify_response(prompt) # Use the new function
                    # Expected answer for evaluation is the ground truth 'incorrect_reasoning_step' description
                    expected = "\n".join(qa.get("incorrect_reasoning_step", [])) # Join list into string
                    f1_score = self.evaluate_open_ended(expected, response)
                    metrics = self.evaluate_open_ended_metrics(expected, response)
                    para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {"question": qa.get("question", ""), "expected": expected, "predicted": response, "f1_score": f1_score, "metrics": metrics}
                    scores.append(f1_score)
                except Exception as e: print(f"Error processing Multi-hop Inverse QA {qa.get('source', {}).get('paragraph_id', 'unknown')}: {e}", flush=True)
            avg_score = sum(scores) / len(scores) if scores else 0.0
            print(f"Average Multi-hop Inverse Score (Semantic): {avg_score:.4f}", flush=True)
            return {"average": avg_score, "scores": results}
        except Exception as e: print(f"Error evaluating Multi-hop Inverse questions: {e}", flush=True); return {"average": 0.0, "scores": {}}

    # --- Overall Execution Function (Unmodified) ---
    def run_all_evaluations(self):
        try:
            if not self.generator_pipeline and not self.verifier_pipeline:
                 print("CRITICAL: No models or pipelines were loaded successfully. Cannot run evaluations.", flush=True)
                 return # Exit if no pipelines loaded

            overall_results = {}
            print("\n--- Starting Evaluations ---", flush=True)
            overall_results["true_false"] = self.evaluate_true_false_questions()
            overall_results["multiple_choice"] = self.evaluate_multiple_choice_questions()
            overall_results["list"] = self.evaluate_list_questions()
            overall_results["short"] = self.evaluate_short_questions()
            overall_results["multi_hop"] = self.evaluate_multi_hop_questions()
            overall_results["short_inverse"] = self.evaluate_short_inverse_questions()
            overall_results["multi_hop_inverse"] = self.evaluate_multi_hop_inverse_questions()
            print("--- Evaluations Complete ---", flush=True)

            # --- Calculate Grand Average Score (Optional) ---
            all_averages = [res.get('average', 0.0) for res in overall_results.values() if isinstance(res, dict)]
            grand_average = sum(all_averages) / len(all_averages) if all_averages else 0.0
            overall_results["GRAND_AVERAGE_SCORE"] = grand_average
            print(f"\n--- Overall Grand Average Score (Simple Mean): {grand_average:.4f} ---", flush=True)
            # Note: Official scoring might use different weighting or metrics.

            output_file = os.path.join(self.base_dir, "overall_evaluation_results.json")
            try:
                with open(output_file, "w") as f:
                    json.dump(overall_results, f, indent=2)
                print(f"Saved overall evaluation results to {output_file}", flush=True)
            except Exception as e_save:
                print(f"Error saving overall results to JSON: {e_save}", flush=True)

        except Exception as e:
            print(f"FATAL Error running overall evaluations: {e}", flush=True)


if __name__ == "__main__":
    print("Initializing Evaluator...", flush=True)
    evaluator = ClinIQLinkSampleDatasetSubmit()
    print("Running Evaluations...", flush=True)
    evaluator.run_all_evaluations()
    print("Script Finished.", flush=True)