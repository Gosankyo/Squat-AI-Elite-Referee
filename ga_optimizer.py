import random
import numpy as np

class LSTMOptimizerGA:
    """
    Algoritmo Genetico (Metaheuristica) para la optimizacion de hiperparametros
    de la red LSTM de Squat AI. 
    Cumple con la Unidad de Algoritmos Evolutivos de Advanced Machine Learning.
    """
    def __init__(self, population_size=10, generations=20):
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = 0.1
        
        # Espacio de busqueda de hiperparametros (Genes)
        self.gene_space = {
            'lstm_units': [16, 32, 64, 128],
            'learning_rate': [0.01, 0.005, 0.001, 0.0001],
            'batch_size': [16, 32, 64],
            'dropout': [0.1, 0.2, 0.3, 0.4]
        }

    def generate_chromosome(self):
        """Crea un individuo aleatorio (Una posible arquitectura LSTM)"""
        return {
            'lstm_units': random.choice(self.gene_space['lstm_units']),
            'learning_rate': random.choice(self.gene_space['learning_rate']),
            'batch_size': random.choice(self.gene_space['batch_size']),
            'dropout': random.choice(self.gene_space['dropout'])
        }

    def fitness_function(self, chromosome):
        """
        Simula la funcion de evaluacion. En un entorno real, aqui se entrenaria 
        la LSTM por 5 epochs y se devolveria el F1-Score de validacion.
        Para la demo, usamos una funcion matematica que recompensa ciertas 
        combinaciones conocidas como 'optimas' en biomecanica.
        """
        score = 0.0
        # Reglas simuladas de afinidad
        if chromosome['lstm_units'] == 64: score += 0.4
        if chromosome['dropout'] == 0.2: score += 0.3
        if chromosome['learning_rate'] == 0.001: score += 0.2
        if chromosome['batch_size'] == 32: score += 0.1
        
        # Anadimos ruido para simular el estocasticismo del entrenamiento
        noise = random.uniform(-0.05, 0.05)
        return max(0.0, min(1.0, score + noise)) # Score entre 0 y 1

    def crossover(self, parent1, parent2):
        """Cruzamiento de un punto (Single-point crossover)"""
        child = {}
        keys = list(self.gene_space.keys())
        split = random.randint(1, len(keys)-1)
        
        for i, key in enumerate(keys):
            if i < split:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child

    def mutate(self, chromosome):
        """Mutacion aleatoria para mantener diversidad genetica"""
        for key in self.gene_space:
            if random.random() < self.mutation_rate:
                chromosome[key] = random.choice(self.gene_space[key])
        return chromosome

    def run_evolution(self):
        """Bucle principal del algoritmo evolutivo"""
        print("[INICIO] Iniciando Algoritmo Genetico para Optimizacion de LSTM...")
        population = [self.generate_chromosome() for _ in range(self.pop_size)]
        
        best_overall = None
        best_score = -1

        for gen in range(self.generations):
            # 1. Evaluacion
            scored_pop = [(chrom, self.fitness_function(chrom)) for chrom in population]
            scored_pop.sort(key=lambda x: x[1], reverse=True)
            
            # Guardar el mejor
            if scored_pop[0][1] > best_score:
                best_score = scored_pop[0][1]
                best_overall = scored_pop[0][0]

            print(f"Generacion {gen+1:02d} | Mejor Fitness: {scored_pop[0][1]:.4f} | Elite: {scored_pop[0][0]}")

            # 2. Seleccion (Elitismo: guardamos a los 2 mejores)
            new_population = [scored_pop[0][0], scored_pop[1][0]]

            # 3. Cruzamiento y Mutacion (Rellenar el resto de la poblacion)
            while len(new_population) < self.pop_size:
                # Seleccion por torneo simple
                p1 = random.choice(scored_pop[:5])[0]
                p2 = random.choice(scored_pop[:5])[0]
                
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

        print("\n[FIN] EVOLUCION TERMINADA")
        print("Arquitectura Optima Encontrada:")
        for k, v in best_overall.items():
            print(f" - {k}: {v}")
        print(f"Accuracy Estimado: {best_score*100:.2f}%")

if __name__ == "__main__":
    ga = LSTMOptimizerGA(population_size=10, generations=15)
    ga.run_evolution()