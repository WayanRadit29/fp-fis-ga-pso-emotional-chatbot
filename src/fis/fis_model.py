# src/fis/fis_model.py

import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl


# Names for inputs (emotions) and output (tone classes)
EMOTION_INPUTS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
TONE_CLASSES = ["empathetic", "friendly", "warm", "calming", "supportive", "informative"]


class FISChatbot:
    """
    Fuzzy Inference System for emotional chatbot tone selection.

    Inputs  : 6 emotion scores in [0, 1]
              (sadness, joy, love, anger, fear, surprise)
    Output  : continuous tone index in [0, 5],
              mapped to 6 tone classes:
              0: empathetic, 1: friendly, 2: warm,
              3: calming, 4: supportive, 5: informative
    """

    def __init__(self):
        # Build the fuzzy control system once in the constructor
        self._build_universe()
        self._build_memberships()
        self._build_rules()
        self.control_system = ctrl.ControlSystem(self.rules)

    def _build_universe(self):
        """Define universes for all antecedents and consequent."""
        # Emotion scores are probabilities in [0, 1]
        self.emotion_universe = np.linspace(0.0, 1.0, 101)

        # Tone index universe: 0..5 (six tone classes)
        self.tone_universe = np.linspace(0.0, 5.0, 301)

        # Create fuzzy variables for the six emotions
        self.emotion_vars = {
            name: ctrl.Antecedent(self.emotion_universe, name)
            for name in EMOTION_INPUTS
        }

        # Create fuzzy variable for the tone output
        self.tone_var = ctrl.Consequent(self.tone_universe, "tone")

    def _build_memberships(self):
        """Define membership functions for all variables.

        For each emotion:
            - low, medium, high (trimf)
        For tone:
            - one membership function per tone class (6 total),
              centered on indices 0..5 in the tone universe.
        """

        # Simple generic trimf for all emotions
        # low:   (0.0, 0.0, 0.5)
        # med:   (0.0, 0.5, 1.0)
        # high:  (0.5, 1.0, 1.0)
        for name, var in self.emotion_vars.items():
            var["low"] = fuzz.trimf(var.universe, [0.0, 0.0, 0.5])
            var["med"] = fuzz.trimf(var.universe, [0.0, 0.5, 1.0])
            var["high"] = fuzz.trimf(var.universe, [0.5, 1.0, 1.0])

        # Output tone membership functions:
        # one trimf per tone class, roughly centered at index 0..5
        for idx, tone_name in enumerate(TONE_CLASSES):
            if idx == 0:
                # First class: (0, 0, 1)
                params = [0.0, 0.0, 1.0]
            elif idx == len(TONE_CLASSES) - 1:
                # Last class: (4, 5, 5)
                params = [len(TONE_CLASSES) - 2, len(TONE_CLASSES) - 1, len(TONE_CLASSES) - 1]
            else:
                # Middle classes: (i-1, i, i+1)
                params = [idx - 1, idx, idx + 1]

            self.tone_var[tone_name] = fuzz.trimf(self.tone_var.universe, params)

    def _build_rules(self):
        """Define the fuzzy rule base for tone selection.

        The idea:
        - High sadness  → empathetic
        - High anger    → calming
        - High fear     → supportive
        - High joy      → friendly
        - High love     → warm
        - High surprise → informative
        Plus some mixed rules to make it more realistic.
        """

        sad = self.emotion_vars["sadness"]
        joy = self.emotion_vars["joy"]
        love = self.emotion_vars["love"]
        anger = self.emotion_vars["anger"]
        fear = self.emotion_vars["fear"]
        surprise = self.emotion_vars["surprise"]
        tone = self.tone_var

        rules = []

        # Core direct rules
        rules.append(ctrl.Rule(sad["high"] & joy["low"], tone["empathetic"]))
        rules.append(ctrl.Rule(sad["med"] & joy["low"], tone["empathetic"]))
        rules.append(ctrl.Rule(joy["high"] & sad["low"], tone["friendly"]))
        rules.append(ctrl.Rule(love["high"], tone["warm"]))
        rules.append(ctrl.Rule(anger["high"] & sad["low"], tone["calming"]))
        rules.append(ctrl.Rule(fear["high"], tone["supportive"]))
        rules.append(ctrl.Rule(surprise["high"] & sad["low"], tone["informative"]))

        # Mixed emotional states
        # When sadness and fear both medium/high → combine empathy and support → supportive
        rules.append(ctrl.Rule(sad["high"] & fear["med"], tone["supportive"]))
        rules.append(ctrl.Rule(sad["med"] & fear["high"], tone["supportive"]))

        # When anger + sadness → calming (de-escalation)
        rules.append(ctrl.Rule(anger["high"] & sad["high"], tone["calming"]))

        # When joy and love both high → warm + friendly → warm
        rules.append(ctrl.Rule(joy["high"] & love["high"], tone["warm"]))

        # When surprise and joy high → informative but still friendly
        rules.append(ctrl.Rule(surprise["med"] & joy["high"], tone["informative"]))

        self.rules = rules

    def _simulate_single(self, x_vec):
        """Run fuzzy inference for a single 6D emotion vector.

        x_vec: array-like of shape (6,)
        returns: continuous tone value in [0,5]
        """
        if len(x_vec) != 6:
            raise ValueError(f"Expected 6 inputs (sadness, joy, love, anger, fear, surprise), got {len(x_vec)}")

        # New simulation per sample to avoid state carry-over
        sim = ctrl.ControlSystemSimulation(self.control_system)

        # Set inputs
        for name, value in zip(EMOTION_INPUTS, x_vec):
            sim.input[name] = float(value)

        # Compute fuzzy inference
        sim.compute()

        tone_value = sim.output["tone"]
        return tone_value

    def predict_one(self, x_vec, return_label=False):
        """Predict tone class for a single 6D input.

        Parameters
        ----------
        x_vec : array-like, shape (6,)
            Emotion scores in [0,1].
        return_label : bool
            If True, return tone label (string).
            If False, return class index (int).

        Returns
        -------
        int or str
            Predicted tone index or tone label.
        """
        tone_value = self._simulate_single(x_vec)

        # Map continuous tone value to discrete class index [0..5]
        tone_idx = int(round(tone_value))
        tone_idx = int(np.clip(tone_idx, 0, len(TONE_CLASSES) - 1))

        if return_label:
            return TONE_CLASSES[tone_idx]
        return tone_idx

    def predict_batch(self, X, return_label=False):
        """Predict tone for a batch of inputs.

        Parameters
        ----------
        X : array-like, shape (n_samples, 6)
            Batch of emotion score vectors.
        return_label : bool
            If True, return list of tone labels.
            If False, return numpy array of tone indices.

        Returns
        -------
        np.ndarray or list
            Predicted classes for each row in X.
        """
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != 6:
            raise ValueError(f"Expected X shape (n_samples, 6), got {X.shape}")

        preds = []
        for row in X:
            preds.append(self.predict_one(row, return_label=return_label))

        if return_label:
            return preds
        return np.array(preds, dtype=int)
