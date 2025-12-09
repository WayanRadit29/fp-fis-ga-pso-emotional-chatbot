import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from sklearn.metrics import f1_score
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from fis.fis_model import FISChatbot, EMOTION_INPUTS, TONE_CLASSES

class OptimizedFISChatbot(FISChatbot):
    """
    FIS Model wrapper yang ringan untuk evaluasi berulang.
    Sama seperti implementasi di GA, memungkinkan injeksi parameter output MF.
    """
    def __init__(self, output_mf_params=None):
        self.output_mf_params = output_mf_params
        super().__init__()
        # Pre-compile simulation untuk efisiensi
        self.sim_instance = ctrl.ControlSystemSimulation(self.control_system)

    def _build_memberships(self):
        """Override: Input MF default, Output MF dinamis dari parameter."""
        # 1. Input MF (Tetap Default/Static)
        for name, var in self.emotion_vars.items():
            var["low"] = fuzz.trimf(var.universe, [0.0, 0.0, 0.5])
            var["med"] = fuzz.trimf(var.universe, [0.0, 0.5, 1.0])
            var["high"] = fuzz.trimf(var.universe, [0.5, 1.0, 1.0])

        # 2. Output MF (Dinamis dari PSO)
        if self.output_mf_params is None:
            # Default initialization jika tidak ada params
            for idx, tone_name in enumerate(TONE_CLASSES):
                if idx == 0: params = [0.0, 0.0, 1.0]
                elif idx == 5: params = [4.0, 5.0, 5.0]
                else: params = [idx - 1, idx, idx + 1]
                self.tone_var[tone_name] = fuzz.trimf(self.tone_var.universe, params)
        else:
            # Injeksi parameter dari partikel PSO
            for tone_name in TONE_CLASSES:
                params = self.output_mf_params[tone_name]
                self.tone_var[tone_name] = fuzz.trimf(self.tone_var.universe, params)

    def _simulate_single(self, x_vec):
        """Versi cepat dengan fail-safe value."""
        for name, value in zip(EMOTION_INPUTS, x_vec):
            self.sim_instance.input[name] = float(value)

        try:
            self.sim_instance.compute()
            return self.sim_instance.output["tone"]
        except:
            # Return nilai tengah (2.5) jika aturan tidak ter-trigger (dead zone)
            return 2.5


class PSOFISOptimizer:
    """
    Particle Swarm Optimization (PSO) yang dioptimasi untuk kecepatan dan stabilitas.
    
    Fitur Optimasi (Diadaptasi dari GA):
    1. Stratified Mini-Batch Sampling: Evaluasi pada subset data yang seimbang.
    2. Smart Initialization: Partikel mulai di dekat solusi logis, bukan random buta.
    3. Vectorized Operations: Update posisi menggunakan operasi matriks NumPy.
    4. Early Stopping: Berhenti jika konvergen.
    """

    def __init__(self, X_train, y_train, batch_size=200, stratified=True):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = min(batch_size, len(X_train))
        self.stratified = stratified
        
        # Setup Stratified Sampling Indices
        if stratified:
            self.class_indices = {}
            unique_classes = np.unique(y_train)
            for cls in unique_classes:
                self.class_indices[cls] = np.where(y_train == cls)[0]
        
        # Konfigurasi Dimensi Partikel
        # 6 tone classes * 3 parameter (a, b, c) per segitiga
        self.n_tones = len(TONE_CLASSES)
        self.params_per_tone = 3
        self.particle_dim = self.n_tones * self.params_per_tone
        
        # Variable untuk menyimpan hasil terbaik
        self.global_best_position = None
        self.global_best_fitness = -1.0
        self.best_params_dict = None
        self.history = []

    def _get_batch_indices(self):
        """Memilih index sampel secara acak namun proporsional (Stratified)."""
        if self.stratified and hasattr(self, 'class_indices'):
            n_classes = len(self.class_indices)
            samples_per_class = max(1, self.batch_size // n_classes)
            
            batch_indices = []
            for cls, indices in self.class_indices.items():
                # Pilih n sampel dari kelas ini
                selected = np.random.choice(indices, min(len(indices), samples_per_class), replace=False)
                batch_indices.extend(selected)
            
            # Shuffle agar urutan kelas tidak kaku
            np.random.shuffle(batch_indices)
            return np.array(batch_indices[:self.batch_size])
        else:
            # Simple random sampling
            return np.random.choice(len(self.X_train), self.batch_size, replace=False)

    def _particle_to_params(self, particle_vector):
        """
        Decode vektor partikel menjadi dictionary parameter valid.
        Memastikan constraint a <= b <= c.
        """
        params = {}
        for i, tone_name in enumerate(TONE_CLASSES):
            start_idx = i * self.params_per_tone
            genes = particle_vector[start_idx : start_idx + 3]
            
            # Constraint Handling 1: Sorting (a <= b <= c)
            a, b, c = np.sort(genes)
            
            # Constraint Handling 2: Clipping (Agar tidak keluar universe 0-5)
            # Diberi sedikit margin agar tidak error di ujung
            a = np.clip(a, 0.0, 4.8)
            b = np.clip(b, a + 0.05, 4.95)
            c = np.clip(c, b + 0.05, 5.0)
            
            params[tone_name] = [float(a), float(b), float(c)]
        return params

    def _calculate_fitness(self, particle, X_batch, y_batch):
        """Evaluasi satu partikel pada mini-batch."""
        param_dict = self._particle_to_params(particle)
        
        try:
            # Init Model dengan parameter dari partikel
            model = OptimizedFISChatbot(output_mf_params=param_dict)
            
            # Prediksi
            y_pred = model.predict_batch(X_batch)
            
            # Hitung Akurasi
            return f1_score(y_batch, y_pred, average='weighted')
        except Exception:
            # Penalty jika konfigurasi parameter menyebabkan error
            return 0.0

    def _smart_initialization(self, n_particles):
        """
        Inisialisasi partikel di sekitar 'Center' ideal.
        Ini mempercepat konvergensi dibanding random total.
        """
        positions = np.zeros((n_particles, self.particle_dim))
        
        for p in range(n_particles):
            for i in range(self.n_tones):
                # Tone index ideal: 0, 1, 2, 3, 4, 5
                center_ideal = i 
                start_idx = i * 3
                
                # Tambahkan noise random di sekitar center ideal
                # a (kiri), b (tengah), c (kanan)
                spread = np.random.uniform(0.4, 1.0) # Lebar segitiga variatif
                
                # Generate a, b, c dengan noise
                noise_b = np.random.uniform(-0.25, 0.25)
                b_val = center_ideal + noise_b
                a_val = b_val - spread
                c_val = b_val + spread
                
                positions[p, start_idx:start_idx+3] = [a_val, b_val, c_val]
                
            # Pastikan valid sejak awal
            positions[p] = np.clip(positions[p], 0.0, 5.0)
            
        return positions

    def run(self, n_particles=20, n_iterations=30, 
            w=0.7, c1=1.5, c2=1.5, 
            early_stopping_rounds=5, verbose=True):
        """
        Jalankan loop optimasi PSO.
        """
        if verbose:
            print(f"PSO Config: Particles={n_particles}, Iter={n_iterations}, Batch={self.batch_size}, Stratified={self.stratified}")

        # 1. Initialization
        positions = self._smart_initialization(n_particles)
        velocities = np.zeros_like(positions)
        
        personal_best_pos = positions.copy()
        personal_best_scores = np.full(n_particles, -1.0)
        
        self.global_best_fitness = -1.0
        self.global_best_position = positions[0].copy()
        
        no_improv_count = 0
        self.history = []

        # 2. Main Loop
        for it in range(n_iterations):
            # A. Ambil Mini-Batch Baru (Stochastic Evaluation)
            # Ini kunci agar running cepat!
            indices = self._get_batch_indices()
            X_batch = self.X_train[indices]
            y_batch = self.y_train[indices]
            
            iter_best_score = -1.0
            
            # B. Evaluasi Partikel
            # (Bisa diparallelkan dengan ThreadPoolExecutor jika perlu, tapi skfuzzy kadang thread-unsafe)
            current_scores = []
            for i in range(n_particles):
                score = self._calculate_fitness(positions[i], X_batch, y_batch)
                current_scores.append(score)
                
                # Update PBest
                if score > personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_pos[i] = positions[i].copy()
            
            current_scores = np.array(current_scores)
            
            # C. Update Global Best
            # Kita bandingkan max score batch ini dengan global best
            # Note: Ada risiko noisy update karena batch berubah, tapi pbest menjaga memori
            max_idx = np.argmax(current_scores)
            max_score = current_scores[max_idx]
            
            if max_score > self.global_best_fitness:
                self.global_best_fitness = max_score
                self.global_best_position = positions[max_idx].copy()
                no_improv_count = 0
                improved = True
            else:
                no_improv_count += 1
                improved = False
            
            self.history.append(self.global_best_fitness)
            
            if verbose and (it + 1) % 5 == 0:
                print(f"  Iter {it+1}/{n_iterations} | Batch Best: {max_score:.4f} | Global Best: {self.global_best_fitness:.4f}")

            # D. Early Stopping
            if early_stopping_rounds and no_improv_count >= early_stopping_rounds:
                if verbose: print(f"  Early stopping at iter {it+1} (no improvement for {early_stopping_rounds} iters)")
                break
                
            # E. Update Position & Velocity (Vectorized)
            r1 = np.random.rand(n_particles, self.particle_dim)
            r2 = np.random.rand(n_particles, self.particle_dim)
            
            velocities = (w * velocities) + \
                         (c1 * r1 * (personal_best_pos - positions)) + \
                         (c2 * r2 * (self.global_best_position - positions))
            
            positions = positions + velocities
            
            # Boundary Handling
            positions = np.clip(positions, 0.0, 5.0)

        # 3. Finalization
        self.best_params_dict = self._particle_to_params(self.global_best_position)
        
        if verbose:
            print(f"PSO Finished. Best Fitness: {self.global_best_fitness:.4f}")
            
        return self.global_best_fitness

    def get_optimized_model(self):
        """Return object OptimizedFISChatbot yang sudah dituning."""
        if self.best_params_dict is None:
            raise ValueError("Run PSO first!")
        return OptimizedFISChatbot(output_mf_params=self.best_params_dict)
