using JuMP, GLPK

model = Model(GLPK.Optimizer)

licores = [:Añejo, :Frutado, :Seco]
meses = 1:4

# Parámetros
precio = Dict(
    (:Añejo, 1)=>50, (:Añejo, 2)=>50, (:Añejo, 3)=>50, (:Añejo, 4)=>50,
    (:Frutado, 1)=>55, (:Frutado, 2)=>55, (:Frutado, 3)=>55, (:Frutado, 4)=>55,
    (:Seco, 1)=>45, (:Seco, 2)=>45, (:Seco, 3)=>45, (:Seco, 4)=>45,
)

demanda_max = Dict(
    (:Añejo, 1)=>10000, (:Añejo, 2)=>12000, (:Añejo, 3)=>14000, (:Añejo, 4)=>8000,
    (:Frutado, 1)=>12000, (:Frutado, 2)=>13000, (:Frutado, 3)=>10000, (:Frutado, 4)=>11000,
    (:Seco, 1)=>11000, (:Seco, 2)=>13000, (:Seco, 3)=>12000, (:Seco, 4)=>9000,
)

insumos = [:A, :B, :C]
disponibilidad = Dict(:A=>6000, :B=>7500, :C=>5600)
costo_insumo = Dict(:A=>35, :B=>25, :C=>20)

# Proporciones
proporcion = Dict(
    (:Añejo, :A)=>0.3, (:Añejo, :B)=>0.2, (:Añejo, :C)=>0.5,
    (:Frutado, :A)=>0.1, (:Frutado, :B)=>0.6, (:Frutado, :C)=>0.3,
    (:Seco, :A)=>0.2, (:Seco, :B)=>0.4, (:Seco, :C)=>0.4,
)

# Variables
@variable(model, x[licores, meses] >= 0)   # Producción
@variable(model, s[licores, 0:4] >= 0)     # Stock
@variable(model, v[licores, meses] >= 0)   # Venta

# Restricción: relación entre producción, stock y venta
for l in licores, m in meses
    if m == 1
        @constraint(model, x[l,m] == v[l,m] + s[l,m])
    else
        @constraint(model, x[l,m] + s[l,m-1] == v[l,m] + s[l,m])
    end
end

# Restricción: demanda máxima
for l in licores, m in meses
    @constraint(model, v[l,m] <= demanda_max[(l,m)])
end

# Restricción: disponibilidad mensual de insumos
for m in meses, i in insumos
    @constraint(model, sum(proporcion[(l,i)] * x[l,m] for l in licores) <= disponibilidad[i])
end

# Función objetivo: maximizar ganancias
@objective(model, Max,
    sum(precio[(l,m)] * v[l,m] for l in licores, m in meses) -
    sum(proporcion[(l,i)] * x[l,m] * costo_insumo[i] for l in licores, i in insumos, m in meses) -
    sum(0.5 * s[l,m] for l in licores, m in meses)
)

optimize!(model)

# Mostrar resultados
println("Estado de optimización: ", termination_status(model))
println("\nProducción óptima por mes:")
for l in licores, m in meses
    println("Producción de $l en mes $m: ", value(x[l,m]))
end

println("\nVentas por mes:")
for l in licores, m in meses
    println("Ventas de $l en mes $m: ", value(v[l,m]))
end

println("\nStock al final de cada mes:")
for l in licores, m in 1:4
    println("Stock de $l al final del mes $m: ", value(s[l,m]))
end

println("\nUtilidad total: ", objective_value(model))
