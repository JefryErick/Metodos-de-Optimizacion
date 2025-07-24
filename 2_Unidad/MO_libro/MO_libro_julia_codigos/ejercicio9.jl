using JuMP, HiGHS

# Conjuntos
profesores = 1:4  # P1, P2, P3, P4
asignaturas = 1:5 # A1 to A5

# Valoraciones (matriz de 4x5)
valoraciones = [
    2.7 2.2 3.4 2.8 3.6;
    2.0 3.6 3.4 2.8 3.6;
    3.2 3.8 2.3 1.9 2.6;
    2.6 2.5 1.8 4.2 3.5
]

# Crear el modelo
model = Model(HiGHS.Optimizer)

# Variables binarias: x[i,j] = 1 si el profesor i enseña la asignatura j
@variable(model, x[profesores, asignaturas], Bin)

# Función objetivo: maximizar la suma ponderada por las valoraciones
@objective(model, Max, sum(valoraciones[i,j] * x[i,j] for i in profesores, j in asignaturas))

# Cada asignatura debe ser asignada a un solo profesor
@constraint(model, [j in asignaturas], sum(x[i,j] for i in profesores) == 1)

# Restricción de carga docente
@constraint(model, sum(x[1,j] for j in asignaturas) <= 1)           # P1
@constraint(model, sum(x[2,j] for j in asignaturas) <= 2)           # P2
@constraint(model, sum(x[3,j] for j in asignaturas) <= 2)           # P3
@constraint(model, sum(x[4,j] for j in asignaturas) <= 2)           # P4

# Restricción: P3 no puede dictar A1 ni A2
@constraint(model, x[3,1] == 0)
@constraint(model, x[3,2] == 0)

# Resolver
optimize!(model)

# Verificar estado
println("Estado de la solución: ", termination_status(model))

# Valor óptimo
println("Valoración total máxima: ", objective_value(model))

# Asignación óptima
for j in asignaturas
    for i in profesores
        if value(x[i,j]) > 0.5  # por tolerancia
            println("Asignatura A$j asignada a Profesor P$i")
        end
    end
end
