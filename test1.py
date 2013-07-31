import random
import csv
from deap import base
from deap import creator
from deap import tools
#Maximiza el Fitness con valores positivos
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def Inicial():
    contents = csv.reader(open("prueba.csv", "r"))
    matriz = list(list(float(elem) for elem in row) for row in contents)    
    return matriz

toolbox = base.Toolbox()

toolbox.register("individual", Inicial)
#Crea la poblacion inicial
toolbox.register("population",toolbox.individual)

def evalOneMax(individual):
    return sum(individual),

# Crea los operadores para el algoritmo genetico
toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoints)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    #Semilla    
    random.seed(100)
    #Poblacion inicial de 100
    pop = toolbox.population()
    print "-------------Crea la poblacion con 10 atributos----------------------"   
    print pop    
    print "-------------FIN DE LA POBLACION----------------------"     
    CXPB, MUTPB, NGEN = 0.5, 0.01, 40
    
    print("*****************INICIA LA EVOLUCION*******************************")
    
    # Evalua la popblacion entera
    fitnesses = list(map(toolbox.evaluate, pop))
    print "---------FITNESSES--------"    
    print fitnesses
    print"------------------------POBLACION-----------------------------------"
    print pop    
    print "--------------------Evaluacion de la poblacion---------------------"    
    print fitnesses    
    for ind, fit in zip(pop, fitnesses):
        print "-----FIT------"        
        print fit        
        ind.fitness.values = fit        
        print "-----FITNESSES------"
        print ind.fitness.values 
    print("--------------------Evalua el %i de forma individual----------------" % len(pop))
    
    # Inicio de la evolucion
    for g in range(NGEN):
        print("-- Generacion # %i --" % g)
        
        # Selecciona la proxima generacion de individuos
        offspring = toolbox.select(pop, len(pop))
        #print "**********Individuos seleccionados*****************************"        
        #print offspring        
        # Clona a los individuos seleccionados offspring
        offspring = list(map(toolbox.clone, offspring))
    
        # Aplica el cruzamiento y la mutacion de los individuos seleccionados
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evalua los individuos con fitness invalido
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print("---------------Evaluacion %i individuos" % len(invalid_ind))
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
    
    #print("--------------------El fin de la (satisfactoria) evolucion ----------------")
    
    best_ind = tools.selBest(pop, 1)[0]
    print("=====================> El mejor individuo es %s, %s" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":
    main()