import sys
import os
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import random
from sklearn.metrics import f1_score
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from fis.fis_model import FISChatbot, EMOTION_INPUTS, TONE_CLASSES

class OptimizedFISChatbot(FISChatbot):
    """
    FIS dengan parameter output MF yang bisa di-inject.
    Input MF tetap default.
    """
    def __init__(self, output_mf_params=None):
        self.output_mf_params = output_mf_params
        super().__init__()

        self.sim_instance = ctrl.ControlSystemSimulation(self.control_system)

    def _build_memberships(self):
        """Override: Input MF default, Output MF dari parameter GA."""
        for name, var in self.emotion_vars.items():
            var["low"] = fuzz.trimf(var.universe, [0.0, 0.0, 0.5])
            var["med"] = fuzz.trimf(var.universe, [0.0, 0.5, 1.0])
            var["high"] = fuzz.trimf(var.universe, [0.5, 1.0, 1.0])

        if self.output_mf_params is None:
            for idx, tone_name in enumerate(TONE_CLASSES):
                if idx == 0:
                    params = [0.0, 0.0, 1.0]
                elif idx == len(TONE_CLASSES) - 1:
                    params = [4.0, 5.0, 5.0]
                else:
                    params = [idx - 1, idx, idx + 1]
                self.tone_var[tone_name] = fuzz.trimf(self.tone_var.universe, params)
        else:
            for tone_name in TONE_CLASSES:
                params = self.output_mf_params[tone_name]
                self.tone_var[tone_name] = fuzz.trimf(self.tone_var.universe, params)

    def _simulate_single(self, x_vec):
        """Override dengan fail-safe."""
        for name, value in zip(EMOTION_INPUTS, x_vec):
            self.sim_instance.input[name] = float(value)

        try:
            self.sim_instance.compute()
            return self.sim_instance.output["tone"]
        except:
            return 2.5


class GAFISOptimizer:
    """
    Genetic Algorithm untuk optimasi OUTPUT Membership Function.
    
    OPTIMIZED VERSION:
    - Mini-batch sampling untuk fitness evaluation
    - Early stopping jika tidak ada improvement
    - Stratified sampling untuk representasi kelas lebih baik
    """
    
    def __init__(self, X_train, y_train, batch_size=200, stratified=True):
        """
        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        batch_size : int
            Ukuran batch untuk fitness evaluation (default: 200)
        stratified : bool
            Gunakan stratified sampling (default: True)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = min(batch_size, len(X_train))
        self.stratified = stratified
        
        # Pre-compute indices per class untuk stratified sampling
        if stratified:
            self.class_indices = {}
            for cls in np.unique(y_train):
                self.class_indices[cls] = np.where(y_train == cls)[0]
        
        # Chromosome: 6 tone classes × 3 params = 18 genes
        self.num_tones = len(TONE_CLASSES)
        self.genes_per_tone = 3
        self.chromosome_length = self.num_tones * self.genes_per_tone
        
        self.best_individual = None
        self.best_fitness = -1.0
        self.best_params = None
        self.history = []

    def _get_batch_indices(self):
        """Get batch indices (stratified or random)."""
        if self.stratified and len(self.class_indices) > 0:
            # Stratified: ambil proporsi yang sama dari setiap kelas
            samples_per_class = max(1, self.batch_size // len(self.class_indices))
            indices = []
            for cls, cls_indices in self.class_indices.items():
                n_samples = min(samples_per_class, len(cls_indices))
                selected = np.random.choice(cls_indices, n_samples, replace=False)
                indices.extend(selected)
            return np.array(indices[:self.batch_size])
        else:
            return np.random.choice(len(self.X_train), self.batch_size, replace=False)

    def _chromosome_to_params(self, chromosome):
        """Decode chromosome ke dictionary parameter output MF."""
        params = {}
        for i, tone_name in enumerate(TONE_CLASSES):
            start_idx = i * self.genes_per_tone
            genes = chromosome[start_idx:start_idx + 3]
            a, b, c = sorted(genes)
            
            # Clip dengan margin minimal
            a = np.clip(a, 0.0, 4.8)
            b = np.clip(b, a + 0.1, 4.9)
            c = np.clip(c, b + 0.1, 5.0)
            
            params[tone_name] = [float(a), float(b), float(c)]
        return params

    def _calculate_fitness(self, chromosome):
        """Hitung fitness pada mini-batch."""
        output_params = self._chromosome_to_params(chromosome)
        
        try:
            fis = OptimizedFISChatbot(output_mf_params=output_params)
        except:
            return 0.0
        
        indices = self._get_batch_indices()
        X_batch = self.X_train[indices]
        y_batch = self.y_train[indices]
        
        try:
            y_pred = fis.predict_batch(X_batch)
            return f1_score(y_batch, y_pred, average='weighted')
        except:
            return 0.0

    def _initialize_population(self, pop_size):
        """Inisialisasi populasi dengan parameter di sekitar center yang sesuai."""
        population = []
        
        for _ in range(pop_size):
            chromosome = np.zeros(self.chromosome_length)
            
            for i in range(self.num_tones):
                center = i
                start_idx = i * self.genes_per_tone
                
                spread = np.random.uniform(0.5, 1.2)
                a = max(0.0, center - spread)
                b = center + np.random.uniform(-0.2, 0.2)
                b = np.clip(b, a + 0.1, 4.9)
                c = min(5.0, center + spread)
                c = max(c, b + 0.1)
                
                chromosome[start_idx:start_idx + 3] = [a, b, c]
            
            population.append(chromosome)
        
        return np.array(population)

    def _tournament_selection(self, population, fitnesses, k=3):
        """Tournament selection."""
        pop_size = len(population)
        k = min(k, pop_size)
        selected_indices = np.random.choice(pop_size, k, replace=False)
        best_idx = selected_indices[np.argmax(fitnesses[selected_indices])]
        return population[best_idx].copy()

    def _crossover(self, parent1, parent2, crossover_rate=0.8):
        """Single Point Crossover."""
        if random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        point = random.randint(1, self.chromosome_length - 1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        
        return child1, child2

    def _mutate(self, individual, mutation_rate=0.15, mutation_strength=0.3):
        """Gaussian Mutation."""
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                noise = np.random.normal(0, mutation_strength)
                mutated[i] += noise
                mutated[i] = np.clip(mutated[i], 0.0, 5.0)
        
        return mutated

    def run(self, num_generations=30, population_size=20, 
            crossover_rate=0.8, mutation_rate=0.15,
            early_stopping_rounds=10, verbose=True):
        """
        Main GA Loop dengan early stopping.
        
        Parameters
        ----------
        num_generations : int
            Jumlah generasi maksimum
        population_size : int
            Ukuran populasi
        crossover_rate : float
            Probabilitas crossover
        mutation_rate : float
            Probabilitas mutasi per gen
        early_stopping_rounds : int
            Hentikan jika tidak ada improvement selama N generasi
        verbose : bool
            Print progress
            
        Returns
        -------
        float : Best fitness score
        """
        if verbose:
            print(f"  GA Config: Pop={population_size}, Gen={num_generations}, "
                  f"Batch={self.batch_size}, Chromosome={self.chromosome_length} genes")
        
        population = self._initialize_population(population_size)
        self.history = []
        no_improvement_count = 0

        for gen in range(num_generations):
            # Evaluate Fitness
            fitnesses = np.array([self._calculate_fitness(ind) for ind in population])
            
            # Track Best
            max_fit_idx = np.argmax(fitnesses)
            current_best_fit = fitnesses[max_fit_idx]
            current_best_ind = population[max_fit_idx]
            
            improved = False
            if current_best_fit > self.best_fitness:
                self.best_fitness = current_best_fit
                self.best_individual = current_best_ind.copy()
                self.best_params = self._chromosome_to_params(self.best_individual)
                improved = True
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            self.history.append({
                'generation': gen + 1,
                'best_fitness': self.best_fitness,
                'avg_fitness': np.mean(fitnesses),
                'current_best': current_best_fit
            })
            
            # Progress
            if verbose and (gen + 1) % 5 == 0:
                print(f"    Gen {gen+1}/{num_generations} | "
                      f"Best: {self.best_fitness:.4f} | "
                      f"Avg: {np.mean(fitnesses):.4f}")
            
            # Early Stopping
            if early_stopping_rounds and no_improvement_count >= early_stopping_rounds:
                if verbose:
                    print(f"    Early stopping at gen {gen+1} (no improvement for {early_stopping_rounds} gens)")
                break
            
            # Elitism
            new_population = [self.best_individual.copy()]
            
            # Selection & Reproduction
            while len(new_population) < population_size:
                p1 = self._tournament_selection(population, fitnesses)
                p2 = self._tournament_selection(population, fitnesses)
                
                c1, c2 = self._crossover(p1, p2, crossover_rate)
                c1 = self._mutate(c1, mutation_rate)
                c2 = self._mutate(c2, mutation_rate)
                
                new_population.append(c1)
                if len(new_population) < population_size:
                    new_population.append(c2)
            
            population = np.array(new_population)

        if verbose:
            print(f"  GA Complete! Best Fitness: {self.best_fitness:.4f}")
        
        return self.best_fitness

    def get_optimized_model(self):
        """Return FIS model dengan parameter optimal."""
        if self.best_params is None:
            raise ValueError("GA belum dijalankan. Panggil .run() terlebih dahulu.")
        return OptimizedFISChatbot(output_mf_params=self.best_params)
    
    def get_best_params(self):
        """Return parameter MF output terbaik."""
        return self.best_params
    
    def get_history(self):
        """Return history fitness per generasi."""
        return self.history
