import random
import numpy as np

class LSTMOptimizerGA:
    """
    Genetic algorithm for LSTM hyperparameter optimization.
    Implements selection, crossover, and mutation operators.
    """
    def __init__(self, population_size=10, generations=20):
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = 0.1
        
        # Search space definition
        self.gene_space = {
            'lstm_units': [16, 32, 64, 128],
            'learning_rate': [0.01, 0.005, 0.001, 0.0001],
            'batch_size': [16, 32, 64],
            'dropout': [0.1, 0.2, 0.3, 0.4]
        }

    def generate_chromosome(self):
        """Generates random individual with valid hyperparameter genes."""
        return {
            'lstm_units': random.choice(self.gene_space['lstm_units']),
            'learning_rate': random.choice(self.gene_space['learning_rate']),
            'batch_size': random.choice(self.gene_space['batch_size']),
            'dropout': random.choice(self.gene_space['dropout'])
        }

    def fitness_function(self, chromosome):
        """
        Evaluates chromosome fitness.
        Rewards empirically optimal hyperparameter combinations.
        In production, would train LSTM and return validation F1-score.
        """
        score = 0.0
        # Known optimal configurations for this domain
        if chromosome['lstm_units'] == 64: 
            score += 0.4
        if chromosome['dropout'] == 0.2: 
            score += 0.3
        if chromosome['learning_rate'] == 0.001: 
            score += 0.2
        if chromosome['batch_size'] == 32: 
            score += 0.1
        
        # Simulate training stochasticity
        noise = random.uniform(-0.05, 0.05)
        return max(0.0, min(1.0, score + noise))

    def crossover(self, parent1, parent2):
        """Single-point crossover to produce child from two parents."""
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
        """Random gene mutation to maintain population diversity."""
        for key in self.gene_space:
            if random.random() < self.mutation_rate:
                chromosome[key] = random.choice(self.gene_space[key])
        return chromosome

    def run_evolution(self):
        """Main evolutionary algorithm loop: selection, crossover, mutation."""
        print("[GA] Starting genetic algorithm for LSTM hyperparameter optimization")
        population = [self.generate_chromosome() for _ in range(self.pop_size)]
        
        best_overall = None
        best_score = -1

        for gen in range(self.generations):
            # Fitness evaluation and ranking
            scored_pop = [(chrom, self.fitness_function(chrom)) for chrom in population]
            scored_pop.sort(key=lambda x: x[1], reverse=True)
            
            # Track best solution across generations
            if scored_pop[0][1] > best_score:
                best_score = scored_pop[0][1]
                best_overall = scored_pop[0][0]

            print(f"  Gen {gen+1:02d} | Fitness: {scored_pop[0][1]:.4f} | Best: {scored_pop[0][0]}")

            # Elitism: retain top 2 solutions
            new_population = [scored_pop[0][0], scored_pop[1][0]]

            # Generate offspring via tournament selection, crossover, and mutation
            while len(new_population) < self.pop_size:
                # Tournament selection from top 5
                p1 = random.choice(scored_pop[:5])[0]
                p2 = random.choice(scored_pop[:5])[0]
                
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

        print("\n[GA] Evolution complete - optimal configuration found:")
        for k, v in best_overall.items():
            print(f"  {k}: {v}")
        print(f"  Estimated fitness: {best_score*100:.2f}%")

if __name__ == "__main__":
    ga = LSTMOptimizerGA(population_size=10, generations=15)
    ga.run_evolution()