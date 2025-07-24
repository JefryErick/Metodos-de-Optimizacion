using JuMP, HiGHS

# Crear el modelo
model = Model(HiGHS.Optimizer)

# Variables de decisión: hectolitros a producir
@variable(model, x1 >= 0)  # Extracto para beber
@variable(model, x2 >= 0)  # Extracto concentrado

# Función objetivo: maximizar ganancia
@objective(model, Max, 100x1 + 200x2)

# Restricciones
@constraint(model, 300x1 + 400x2 <= 60000)  # Fruta disponible
@constraint(model, 300x1 + 200x2 <= 48000)  # Preservante disponible
@constraint(model, x1 + 3x2 >= 210)         # Horas mínimas de trabajo
@constraint(model, x2 <= 2x1)               # Proporción entre productos

# Resolver
optimize!(model)

# Mostrar resultados
println("Estado del modelo: ", termination_status(model))
println("Ganancia total: \$", round(objective_value(model), digits=2))
println("Producción óptima:")
println("  Extracto para beber (X₁): ", round(value(x1), digits=3), " hl")
println("  Extracto concentrado (X₂): ", round(value(x2), digits=3), " hl")
