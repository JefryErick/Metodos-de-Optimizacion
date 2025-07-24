using JuMP
using GLPK  # o HiGHS si prefieres

# Crear modelo
model = Model(GLPK.Optimizer)

# Variables de decisión
@variable(model, x[1:3] >= 0, Int)     # x[1]: P1, x[2]: P2, x[3]: P3
@variable(model, y[1:2], Bin)          # y[1]: A y B, y[2]: C y D

# Parámetro M (suficientemente grande)
M = 1_000_000

# Función objetivo: maximizar beneficios netos
@objective(model, Max,
    5 * x[1] + 5 * x[2] + 10 * x[3] - (45 * y[1] + 50 * y[2])
)

# Restricciones por máquinas activas
@constraint(model, x[1] + x[2] + 2x[3] <= 190 + M * (1 - y[1]))  # Máquina A
@constraint(model, x[1] + x[2] + x[3]  <= 210 + M * (1 - y[1]))  # Máquina B
@constraint(model, 2x[1] + x[2] + x[3] <= 170 + M * (1 - y[2]))  # Máquina C
@constraint(model, x[1] + 2x[2] + x[3] <= 200 + M * (1 - y[2]))  # Máquina D

# Solo se puede usar un grupo de máquinas
@constraint(model, y[1] + y[2] == 1)

# Resolver
optimize!(model)

# Mostrar resultados
println("Estado de solución: ", termination_status(model))
println("Beneficio máximo: ", objective_value(model))

for i in 1:3
    println("x_$i = ", value(x[i]))
end

println("y_1 (A y B): ", value(y[1]))
println("y_2 (C y D): ", value(y[2]))
