using JuMP
using HiGHS

# Índices
plantas = 1:4
almacenes = 1:3

# Costos de transporte c[i,j]
costos = [
    3 2 4;
    2 4 3;
    3 5 3;
    4 3 2
]

# Capacidades de plantas
capacidad = [950, 1150, 1000, 900]

# Demandas de almacenes
demanda = [1200, 900, 500]

# Crear modelo
model = Model(HiGHS.Optimizer)

# Variables: x[i,j] ≥ 0
@variable(model, x[plantas, almacenes] >= 0)

# Función objetivo: minimizar el costo total
@objective(model, Min, sum(costos[i,j] * x[i,j] for i in plantas, j in almacenes))

# Restricciones de capacidad por planta
for i in plantas
    @constraint(model, sum(x[i,j] for j in almacenes) <= capacidad[i])
end

# Restricciones de demanda por almacén
for j in almacenes
    @constraint(model, sum(x[i,j] for i in plantas) >= demanda[j])
end

# Resolver
optimize!(model)

# Mostrar resultados
println("Costo total mínimo: \$", objective_value(model))

println("\nAsignaciones óptimas de transporte (x[i,j]):")
for i in plantas
    for j in almacenes
        valor = value(x[i,j])
        if valor > 1e-6
            println("Planta $i → Almacén $j: ", round(valor; digits=2))
        end
    end
end
