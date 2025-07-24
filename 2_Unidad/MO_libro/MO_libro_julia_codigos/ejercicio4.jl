using JuMP
using GLPK

# ================================
# Datos del problema
# ================================
productos = [:A, :B, :C]
dias = 1:6

# Demanda diaria por producto y día
demanda = Dict(
    :A => [45, 30, 35, 25, 30, 35],
    :B => [15, 10, 15, 20, 15, 15],
    :C => [20, 25, 30, 25, 25, 30]
)

# Ritmos de producción (unidades por hora)
ritmo = Dict(:A => 10, :B => 15, :C => 20)

# Inventarios iniciales
inv_inicial = Dict(:A => 60, :B => 0, :C => 10)

# Costos de inventario
costo_inv = Dict(:A => 3, :B => 2, :C => 2)

# Costo por preparación
costo_prep = 150

# Horas disponibles por día
horas_por_dia = 16

# M suficientemente grande
M = 1000

# ================================
# Modelo
# ================================
model = Model(GLPK.Optimizer)

# Variables
@variable(model, x[p in productos, d in dias] >= 0, Int)     # unidades producidas
@variable(model, inv[p in productos, d in dias] >= 0, Int)   # inventario final
@variable(model, z[p in productos, d in dias], Bin)          # preparación

# ================================
# Función objetivo: minimizar costos
# ================================
@objective(model, Min,
    sum(costo_prep * z[p, d] + costo_inv[p] * inv[p, d] for p in productos, d in dias)
)

# ================================
# Restricciones
# ================================

# Relación producción y preparación
@constraint(model, [p in productos, d in dias], x[p, d] <= M * z[p, d])

# Restricciones de capacidad diaria
@constraint(model, [d in dias],
    sum(x[p, d] / ritmo[p] for p in productos) + 2 * sum(z[p, d] for p in productos) <= horas_por_dia
)

# Inventario día 1
@constraint(model, [p in productos],
    inv[p, 1] == x[p, 1] + inv_inicial[p] - demanda[p][1]
)

# Balance de inventario para días siguientes
@constraint(model, [p in productos, d in 2:6],
    inv[p, d] == inv[p, d - 1] + x[p, d] - demanda[p][d]
)

# ================================
# Resolución
# ================================
optimize!(model)

# ================================
# Resultados
# ================================

println("Estado: ", termination_status(model))
println("Costo total mínimo: \$", objective_value(model))

println("\nProducción (x[i,j]) y preparaciones (z[i,j]):")
for p in productos
    println("\nProducto $p:")
    for d in dias
        prod = Int(round(value(x[p, d])))
        prep = Int(round(value(z[p, d])))
        println("  Día $d: $prod unidades (preparación = $prep)")
    end
end

println("\nInventarios finales:")
for p in productos
    println("\nProducto $p:")
    for d in dias
        inv_final = Int(round(value(inv[p, d])))
        println("  Día $d: $inv_final unidades")
    end
end
