using JuMP
using HiGHS

# Crear el modelo
model = Model(HiGHS.Optimizer)

# Variables: número de bloques a producir de cada tipo
@variable(model, x1 >= 100)  # Tipo I
@variable(model, x2 >= 100)  # Tipo II
@variable(model, x3 >= 100)  # Tipo III

# Función objetivo: maximizar utilidad
@objective(model, Max, 6x1 + 8x2 + 9x3)

# Restricciones de recursos
@constraint(model, 1.50x1 + 1.20x2 + 0.80x3 <= 12000)  # Cemento
@constraint(model, 0.80x1 + 0.60x2 + 1.00x3 <= 8000)   # Arena
@constraint(model, 0.40x1 + 0.60x2 + 0.80x3 <= 600)    # Grava
@constraint(model, 0.30x1 + 0.40x2 + 0.50x3 <= 400)    # Agua
@constraint(model, 0.004x1 + 0.002x2 + 0.010x3 <= 300) # Horas máquina

# Resolver el modelo
optimize!(model)

# Mostrar resultados
println("Estado del modelo: ", termination_status(model))
println("Utilidad máxima: \$", objective_value(model))
println("Bloques a producir:")
println("Tipo I: ", value(x1))
println("Tipo II: ", value(x2))
println("Tipo III: ", value(x3))
