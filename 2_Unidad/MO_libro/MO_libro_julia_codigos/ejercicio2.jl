using JuMP
using GLPK  # Puedes cambiarlo por HiGHS si prefieres: using HiGHS

# Crear el modelo
model = Model(GLPK.Optimizer)

# Conjuntos y parámetros
productos = [:pantalon, :chaleco, :chamarra]

piel = Dict(:pantalon => 5, :chaleco => 3, :chamarra => 8)
manobra = Dict(:pantalon => 4, :chaleco => 3, :chamarra => 5)
costo_variable = Dict(:pantalon => 30, :chaleco => 20, :chamarra => 80)
costo_fijo = Dict(:pantalon => 100, :chaleco => 80, :chamarra => 150)
precio_venta = Dict(:pantalon => 60, :chaleco => 40, :chamarra => 120)
min_produccion = Dict(:pantalon => 100, :chaleco => 150, :chamarra => 200)

# Variables de decisión
@variable(model, x[i in productos] >= 0, Int)  # Cantidad a producir (enteras)
@variable(model, y[i in productos], Bin)       # 1 si se produce, 0 si no

# Restricciones
@constraint(model, sum(piel[i]*x[i] for i in productos) <= 3000)      # Piel
@constraint(model, sum(manobra[i]*x[i] for i in productos) <= 2500)   # Mano de obra

# Lógica de lotes: si no se produce, x debe ser 0; si se produce, al menos la mínima
for i in productos
    @constraint(model, x[i] <= 999999 * y[i])
    @constraint(model, x[i] >= min_produccion[i] * y[i])
end

# Función objetivo: maximizar utilidad
@objective(model, Max, sum(
    (precio_venta[i] - costo_variable[i]) * x[i] - costo_fijo[i] * y[i]
    for i in productos)
)

# Resolver el modelo
optimize!(model)

# Mostrar resultados
println("Estado de solución: ", termination_status(model))
println("Utilidad total: \$", objective_value(model))

for i in productos
    println("$i: ", value(x[i]), " unidades (y = ", value(y[i]), ")")
end
