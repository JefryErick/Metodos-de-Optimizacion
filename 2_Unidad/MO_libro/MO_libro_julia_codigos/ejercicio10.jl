using JuMP
using GLPK

# Datos
demanda = [1000, 900, 850, 500, 1000, 600, 1000, 500, 1000]
costo = [20, 20, 20, 21, 21, 21, 22, 22, 22]
n_meses = length(demanda)

# Crear modelo
model = Model(GLPK.Optimizer)

# Variables de decisión
@variable(model, 0 <= X[1:n_meses] <= 1500)  # Compras
@variable(model, Y[1:n_meses] >= 0)          # Inventarios

# Función objetivo
@objective(model, Min, sum(costo[i]*X[i] + 0.2*Y[i] for i in 1:n_meses))

# Restricciones de inventario
@constraint(model, Y[1] == X[1] - demanda[1])
@constraint(model, [i in 2:n_meses], Y[i] == Y[i-1] + X[i] - demanda[i])

# Resolver
optimize!(model)

# Mostrar resultados
println("Costo total mínimo: ", objective_value(model))
for i in 1:n_meses
    println("Mes $i: Comprar = ", value(X[i]), ", Almacenar = ", value(Y[i]))
end
