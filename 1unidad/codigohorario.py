import tkinter as tk
from tkinter import ttk, messagebox
from fpdf import FPDF
from datetime import datetime, timedelta
import re
from collections import defaultdict

# Importar OR-Tools
from ortools.sat.python import cp_model


class RestriccionesHorario:
    def __init__(self):
        self.horas_semanales_totales = 30
        self.hora_inicio_manana = "08:00"
        self.hora_fin_manana = "12:00"
        self.hora_inicio_tarde = "14:00"
        self.hora_fin_tarde = "18:00"
        self.turno_tarde_activo = False  # Nueva variable para controlar turno tarde
        self.max_horas_diarias_por_curso = 3
        self.dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
        self.receso_manana_activo = False
        self.receso_manana_inicio = "10:00"
        self.receso_manana_fin = "10:30"
        self.receso_tarde_activo = False  # Nuevo receso para la tarde
        self.receso_tarde_inicio = "16:00"
        self.receso_tarde_fin = "16:30"
        self.cursos_pesados_nombres = []
        self.umbral_horas_curso_pesado = 6
        self.max_dias_curso_pesado = 3
        self.max_dias_curso_normal = 3
        self.evitar_mismo_dia_cursos_pesados = True

    def validar_horario_general(self, hora_inicio, hora_fin):
        try:
            inicio = datetime.strptime(hora_inicio, "%H:%M")
            fin = datetime.strptime(hora_fin, "%H:%M")
            return fin > inicio
        except ValueError:
            return False

    def validar_receso(self, inicio_receso, fin_receso, hora_inicio_clases, hora_fin_clases):
        try:
            inicio_clases_dt = datetime.strptime(hora_inicio_clases, "%H:%M")
            fin_clases_dt = datetime.strptime(hora_fin_clases, "%H:%M")
            receso_ini_dt = datetime.strptime(inicio_receso, "%H:%M")
            receso_fin_dt = datetime.strptime(fin_receso, "%H:%M")
            return (inicio_clases_dt <= receso_ini_dt < receso_fin_dt <= fin_clases_dt)
        except ValueError:
            return False

    def identificar_cursos_pesados(self, cursos_data): # cursos_data es lista de dicts {'nombre': str, 'horas': int}
        self.cursos_pesados_nombres = [c['nombre'] for c in cursos_data if c['horas_bloques'] > (self.umbral_horas_curso_pesado * (60 // self.duracion_bloque_minutos))]
        # print(f"Cursos pesados identificados: {self.cursos_pesados_nombres}")
        return self.cursos_pesados_nombres

    def set_duracion_bloque(self, duracion_minutos):
        self.duracion_bloque_minutos = duracion_minutos

    # Las validaciones de asignación y globales se manejarán ahora por el solver,
    # pero estas funciones pueden ser útiles para validaciones previas o configuraciones.
    def validar_asignacion_curso(self, curso, dia, horario): # Esta función ya no será usada por el solver directamente
        # ... (lógica original si se necesita para alguna otra cosa) ...
        return True

    def verificar_restricciones_globales(self, cursos_data, horas_semanales_colegio, max_horas_dia_curso, duracion_bloque_min):
        total_horas_cursos_bloques = sum(c['horas_bloques'] for c in cursos_data)
        horas_semanales_colegio_bloques = horas_semanales_colegio * (60 // duracion_bloque_min)

        if total_horas_cursos_bloques > horas_semanales_colegio_bloques:
            return False, f"Total de bloques de cursos ({total_horas_cursos_bloques}) excede el máximo semanal del colegio ({horas_semanales_colegio_bloques})"

        max_bloques_diarios_curso_param = max_horas_dia_curso * (60 // duracion_bloque_min)

        for curso in cursos_data:
            # Esta validación es más compleja con el solver, ya que él mismo encontrará si es posible.
            # Podríamos mantener una validación simple:
            if curso['horas_bloques'] > len(self.dias_semana) * max_bloques_diarios_curso_param:
                 return False, f"El curso {curso['nombre']} requiere más bloques ({curso['horas_bloques']}) de los que se pueden asignar teóricamente."
        return True, "Validaciones preliminares globales OK."

class OptimizadorHorarios(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Optimizador de Horarios Escolares (con OR-Tools)")
        self.geometry("650x700")  # Aumentado para los nuevos controles
        self.resizable(False, False)
        self.restricciones = RestriccionesHorario()
        self.duracion_bloque_var = tk.IntVar(value=60)
        
        self.style = ttk.Style()
        self.style.configure('TLabel', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('TEntry', font=('Arial', 10))
        
        self.cursos_var = tk.IntVar(value=3)
        self.horas_semanales_var = tk.IntVar(value=30)
        self.receso_manana_var = tk.BooleanVar(value=True)
        self.receso_tarde_var = tk.BooleanVar(value=True)
        self.max_horas_diarias_var = tk.IntVar(value=3)
        self.turno_tarde_var = tk.BooleanVar(value=False)  # Nueva variable para turno tarde
        
        self.create_main_frame()
        
    def create_main_frame(self):
        self.main_frame = ttk.Frame(self, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(self.main_frame, text="Número de Cursos:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(self.main_frame, textvariable=self.cursos_var, width=5).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(self.main_frame, text="Horas Semanales del Colegio:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(self.main_frame, textvariable=self.horas_semanales_var, width=5).grid(row=1, column=1, sticky=tk.W)
        
        # Sección de horario mañana
        ttk.Label(self.main_frame, text="Horario Mañana", font=('Arial', 10, 'bold')).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        ttk.Label(self.main_frame, text="Hora Inicio Mañana:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.hora_inicio_manana_h = ttk.Entry(self.main_frame, width=3)
        self.hora_inicio_manana_h.insert(0, "08")
        self.hora_inicio_manana_h.grid(row=3, column=1, sticky=tk.W)
        ttk.Label(self.main_frame, text=":").grid(row=3, column=1, sticky=tk.W, padx=30)
        self.hora_inicio_manana_m = ttk.Entry(self.main_frame, width=3)
        self.hora_inicio_manana_m.insert(0, "00")
        self.hora_inicio_manana_m.grid(row=3, column=1, sticky=tk.W, padx=40)
        
        ttk.Label(self.main_frame, text="Hora Fin Mañana:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.hora_fin_manana_h = ttk.Entry(self.main_frame, width=3)
        self.hora_fin_manana_h.insert(0, "12")
        self.hora_fin_manana_h.grid(row=4, column=1, sticky=tk.W)
        ttk.Label(self.main_frame, text=":").grid(row=4, column=1, sticky=tk.W, padx=30)
        self.hora_fin_manana_m = ttk.Entry(self.main_frame, width=3)
        self.hora_fin_manana_m.insert(0, "00")
        self.hora_fin_manana_m.grid(row=4, column=1, sticky=tk.W, padx=40)

        # Checkbox para activar turno tarde
        ttk.Checkbutton(self.main_frame, text="¿Activar turno tarde?", variable=self.turno_tarde_var,
                       command=self.toggle_turno_tarde).grid(row=5, column=0, columnspan=2, pady=5, sticky=tk.W)
        
        # Sección de horario tarde (inicialmente deshabilitada)
        self.tarde_frame = ttk.Frame(self.main_frame)
        self.tarde_frame.grid(row=6, column=0, columnspan=2, pady=5, sticky=tk.W)
        
        ttk.Label(self.tarde_frame, text="Horario Tarde", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        ttk.Label(self.tarde_frame, text="Hora Inicio Tarde:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.hora_inicio_tarde_h = ttk.Entry(self.tarde_frame, width=3)
        self.hora_inicio_tarde_h.insert(0, "14")
        self.hora_inicio_tarde_h.grid(row=1, column=1, sticky=tk.W)
        ttk.Label(self.tarde_frame, text=":").grid(row=1, column=1, sticky=tk.W, padx=30)
        self.hora_inicio_tarde_m = ttk.Entry(self.tarde_frame, width=3)
        self.hora_inicio_tarde_m.insert(0, "00")
        self.hora_inicio_tarde_m.grid(row=1, column=1, sticky=tk.W, padx=40)
        
        ttk.Label(self.tarde_frame, text="Hora Fin Tarde:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.hora_fin_tarde_h = ttk.Entry(self.tarde_frame, width=3)
        self.hora_fin_tarde_h.insert(0, "18")
        self.hora_fin_tarde_h.grid(row=2, column=1, sticky=tk.W)
        ttk.Label(self.tarde_frame, text=":").grid(row=2, column=1, sticky=tk.W, padx=30)
        self.hora_fin_tarde_m = ttk.Entry(self.tarde_frame, width=3)
        self.hora_fin_tarde_m.insert(0, "00")
        self.hora_fin_tarde_m.grid(row=2, column=1, sticky=tk.W, padx=40)

        # Duración de bloque y máximo horas/día
        ttk.Label(self.main_frame, text="Duración Bloque (min):").grid(row=7, column=0, sticky=tk.W, pady=5)
        ttk.Entry(self.main_frame, textvariable=self.duracion_bloque_var, width=5).grid(row=7, column=1, sticky=tk.W)
        
        ttk.Label(self.main_frame, text="Máx. horas/día por curso:").grid(row=8, column=0, sticky=tk.W, pady=5)
        ttk.Entry(self.main_frame, textvariable=self.max_horas_diarias_var, width=5).grid(row=8, column=1, sticky=tk.W)
        
        # Receso mañana
        ttk.Checkbutton(self.main_frame, text="¿Incluir receso mañana?", variable=self.receso_manana_var, 
                       command=lambda: self.toggle_receso(self.receso_manana_var, self.receso_manana_frame)).grid(row=9, column=0, columnspan=2, pady=5, sticky=tk.W)
        
        self.receso_manana_frame = ttk.Frame(self.main_frame)
        self.receso_manana_frame.grid(row=10, column=0, columnspan=2, pady=5, sticky=tk.W)
        
        ttk.Label(self.receso_manana_frame, text="Inicio Receso Mañana:").grid(row=0, column=0, padx=(0,5))
        self.receso_manana_inicio_h = ttk.Entry(self.receso_manana_frame, width=3)
        self.receso_manana_inicio_h.insert(0, "10")
        self.receso_manana_inicio_h.grid(row=0, column=1)
        ttk.Label(self.receso_manana_frame, text=":").grid(row=0, column=2)
        self.receso_manana_inicio_m = ttk.Entry(self.receso_manana_frame, width=3)
        self.receso_manana_inicio_m.insert(0, "00")
        self.receso_manana_inicio_m.grid(row=0, column=3)

        ttk.Label(self.receso_manana_frame, text="Fin Receso Mañana:").grid(row=0, column=4, padx=(10,5))
        self.receso_manana_fin_h = ttk.Entry(self.receso_manana_frame, width=3)
        self.receso_manana_fin_h.insert(0, "10")
        self.receso_manana_fin_h.grid(row=0, column=5)
        ttk.Label(self.receso_manana_frame, text=":").grid(row=0, column=6)
        self.receso_manana_fin_m = ttk.Entry(self.receso_manana_frame, width=3)
        self.receso_manana_fin_m.insert(0, "30")
        self.receso_manana_fin_m.grid(row=0, column=7)
        
        # Receso tarde (solo visible si turno tarde está activo)
        ttk.Checkbutton(self.main_frame, text="¿Incluir receso tarde?", variable=self.receso_tarde_var, 
                       command=lambda: self.toggle_receso(self.receso_tarde_var, self.receso_tarde_frame)).grid(row=11, column=0, columnspan=2, pady=5, sticky=tk.W)
        
        self.receso_tarde_frame = ttk.Frame(self.main_frame)
        self.receso_tarde_frame.grid(row=12, column=0, columnspan=2, pady=5, sticky=tk.W)
        
        ttk.Label(self.receso_tarde_frame, text="Inicio Receso Tarde:").grid(row=0, column=0, padx=(0,5))
        self.receso_tarde_inicio_h = ttk.Entry(self.receso_tarde_frame, width=3)
        self.receso_tarde_inicio_h.insert(0, "16")
        self.receso_tarde_inicio_h.grid(row=0, column=1)
        ttk.Label(self.receso_tarde_frame, text=":").grid(row=0, column=2)
        self.receso_tarde_inicio_m = ttk.Entry(self.receso_tarde_frame, width=3)
        self.receso_tarde_inicio_m.insert(0, "00")
        self.receso_tarde_inicio_m.grid(row=0, column=3)

        ttk.Label(self.receso_tarde_frame, text="Fin Receso Tarde:").grid(row=0, column=4, padx=(10,5))
        self.receso_tarde_fin_h = ttk.Entry(self.receso_tarde_frame, width=3)
        self.receso_tarde_fin_h.insert(0, "16")
        self.receso_tarde_fin_h.grid(row=0, column=5)
        ttk.Label(self.receso_tarde_frame, text=":").grid(row=0, column=6)
        self.receso_tarde_fin_m = ttk.Entry(self.receso_tarde_frame, width=3)
        self.receso_tarde_fin_m.insert(0, "30")
        self.receso_tarde_fin_m.grid(row=0, column=7)
        
        # Configurar estado inicial
        self.toggle_turno_tarde()
        self.toggle_receso(self.receso_manana_var, self.receso_manana_frame)
        self.toggle_receso(self.receso_tarde_var, self.receso_tarde_frame)

        # Validación de horas
        val_hora = self.register(self.validar_hora_minuto)
        for entry in [self.hora_inicio_manana_h, self.hora_inicio_manana_m, 
                     self.hora_fin_manana_h, self.hora_fin_manana_m,
                     self.hora_inicio_tarde_h, self.hora_inicio_tarde_m,
                     self.hora_fin_tarde_h, self.hora_fin_tarde_m,
                     self.receso_manana_inicio_h, self.receso_manana_inicio_m,
                     self.receso_manana_fin_h, self.receso_manana_fin_m,
                     self.receso_tarde_inicio_h, self.receso_tarde_inicio_m,
                     self.receso_tarde_fin_h, self.receso_tarde_fin_m]:
            entry.config(validate="key", validatecommand=(val_hora, '%P'))
            entry.bind('<FocusOut>', self.formatear_hora_minuto)
        
        ttk.Button(self.main_frame, text="Crear Entradas de Cursos", 
                 command=self.crear_entradas).grid(row=13, column=0, columnspan=2, pady=20)

    def toggle_turno_tarde(self):
        state = "normal" if self.turno_tarde_var.get() else "disabled"
        for widget in self.tarde_frame.winfo_children():
            widget.config(state=state)
        
        # También deshabilitar receso tarde si no hay turno tarde
        if not self.turno_tarde_var.get():
            self.receso_tarde_var.set(False)
            self.toggle_receso(self.receso_tarde_var, self.receso_tarde_frame)
        
        # Habilitar/deshabilitar el checkbox de receso tarde
        self.receso_tarde_frame.grid_remove() if not self.turno_tarde_var.get() else self.receso_tarde_frame.grid()
        for child in self.receso_tarde_frame.winfo_children():
            child.config(state="disabled" if not self.turno_tarde_var.get() else "normal")

    def toggle_receso(self, receso_var, receso_frame):
        state = "normal" if receso_var.get() else "disabled"
        for entry in receso_frame.winfo_children():
            if isinstance(entry, ttk.Entry):
                entry.config(state=state)

    def crear_entradas(self):
        """Crea la ventana para ingresar los datos de los cursos"""
        if not self.validar_horas_colegio():
            return
            
        num_cursos = self.cursos_var.get()
        if num_cursos <= 0:
            messagebox.showerror("Error", "El número de cursos debe ser positivo.")
            return

        self.ventana_cursos = tk.Toplevel(self)
        self.ventana_cursos.title("Ingresar Cursos")
        self.ventana_cursos.geometry("400x300")

        container = ttk.Frame(self.ventana_cursos)
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        container.pack(fill="both", expand=True, padx=10, pady=10)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        ttk.Label(scrollable_frame, text="Nombre Curso", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(scrollable_frame, text="Horas Semanales", font=('Arial', 10, 'bold')).grid(row=0, column=1, padx=5, pady=5)
        
        self.cursos_entries = []
        for i in range(num_cursos):
            ttk.Label(scrollable_frame, text=f"Curso {i + 1}:").grid(row=i+1, column=0, padx=5, pady=2, sticky=tk.E)
            nombre = ttk.Entry(scrollable_frame, width=20)
            nombre.grid(row=i+1, column=1, padx=5, pady=2)
            horas = ttk.Entry(scrollable_frame, width=5)
            horas.grid(row=i+1, column=2, padx=5, pady=2)
            self.cursos_entries.append({'nombre_entry': nombre, 'horas_entry': horas})
        
        button_frame = ttk.Frame(self.ventana_cursos)
        button_frame.pack(fill=tk.X, pady=10, padx=10)
        
        ttk.Button(button_frame, text="Generar Horario", 
                 command=self.procesar_y_generar_horario).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Cancelar", 
                 command=self.ventana_cursos.destroy).pack(side=tk.RIGHT, padx=10)

    def procesar_y_generar_horario(self):
        """Procesa los datos de los cursos y genera el horario"""
        cursos_data = []
        for i, entry_dict in enumerate(self.cursos_entries):
            nombre = entry_dict['nombre_entry'].get().strip()
            horas_str = entry_dict['horas_entry'].get().strip()
            if not nombre:
                messagebox.showerror("Error de Datos", f"El curso {i+1} no tiene nombre.")
                return
            if not horas_str.isdigit() or int(horas_str) <= 0:
                messagebox.showerror("Error de Datos", f"Las horas para '{nombre}' deben ser un número positivo.")
                return
            cursos_data.append({'id': i, 'nombre': nombre, 'horas_semanales': int(horas_str)})

        total_horas_cursos = sum(c['horas_semanales'] for c in cursos_data)
        horas_semanales_colegio = self.horas_semanales_var.get()
        if total_horas_cursos > horas_semanales_colegio:
            messagebox.showerror("Error de Horas",
                f"Las horas totales de los cursos ({total_horas_cursos}) superan las horas semanales del colegio ({horas_semanales_colegio}).")
            return
        
        self.restricciones.set_duracion_bloque(self.duracion_bloque_var.get())
        
        for curso in cursos_data:
            curso['horas_bloques'] = curso['horas_semanales'] * (60 // self.duracion_bloque_var.get())

        valido, mensaje = self.restricciones.verificar_restricciones_globales(
            cursos_data,
            self.horas_semanales_var.get(),
            self.max_horas_diarias_var.get(),
            self.duracion_bloque_var.get()
        )
        if not valido:
            messagebox.showerror("Error de Configuración", mensaje)
            return

        self.restricciones.identificar_cursos_pesados(cursos_data)

        try:
            self.generar_horario_con_optimizacion(cursos_data)
        except Exception as e:
            messagebox.showerror("Error de Optimización", f"Ocurrió un error al generar el horario: {str(e)}\nVerifique la consola para más detalles.")
            print(f"Detalle del error de optimización: {e}")
            import traceback
            traceback.print_exc()

    # ... (los métodos validar_hora_minuto y formatear_hora_minuto se mantienen igual)
    def validar_hora_minuto(self, texto):   
        """Valida que el texto ingresado sea un número válido para horas/minutos"""
        if not texto:  # Permite campo vacío temporalmente
            return True
        if not texto.isdigit():
            return False
        valor = int(texto)
        # Determinar si estamos validando horas o minutos
        widget = self.focus_get()
        if widget in [self.hora_inicio_manana_h, self.hora_fin_manana_h, 
                     self.hora_inicio_tarde_h, self.hora_fin_tarde_h,
                     self.receso_manana_inicio_h, self.receso_manana_fin_h,
                     self.receso_tarde_inicio_h, self.receso_tarde_fin_h]:
            # Validación para horas (0-23)
            return 0 <= valor <= 23
        else:
            # Validación para minutos (0-59)
            return 0 <= valor <= 59

    def formatear_hora_minuto(self, event):
        """Formatea automáticamente las horas y minutos a 2 dígitos"""
        entry = event.widget
        texto = entry.get()
        if not texto:
            return
            
        try:
            valor = int(texto)
            # Determinar si es hora o minuto
            if entry in [self.hora_inicio_manana_h, self.hora_fin_manana_h, 
                        self.hora_inicio_tarde_h, self.hora_fin_tarde_h,
                        self.receso_manana_inicio_h, self.receso_manana_fin_h,
                        self.receso_tarde_inicio_h, self.receso_tarde_fin_h]:
                # Es una hora
                if valor < 0:
                    entry.delete(0, tk.END)
                    entry.insert(0, "00")
                elif valor > 23:
                    entry.delete(0, tk.END)
                    entry.insert(0, "23")
                elif len(texto) == 1:
                    entry.delete(0, tk.END)
                    entry.insert(0, f"0{valor}")
            else:
                # Es un minuto
                if valor < 0:
                    entry.delete(0, tk.END)
                    entry.insert(0, "00")
                elif valor > 59:
                    entry.delete(0, tk.END)
                    entry.insert(0, "59")
                elif len(texto) == 1:
                    entry.delete(0, tk.END)
                    entry.insert(0, f"0{valor}")
        except ValueError:
            # En caso de error, establecer valor por defecto
            if entry in [self.hora_inicio_manana_h, self.hora_fin_manana_h, 
                        self.hora_inicio_tarde_h, self.hora_fin_tarde_h,
                        self.receso_manana_inicio_h, self.receso_manana_fin_h,
                        self.receso_tarde_inicio_h, self.receso_tarde_fin_h]:
                entry.delete(0, tk.END)
                entry.insert(0, "08" if "inicio" in str(entry) else "14")
            else:
                entry.delete(0, tk.END)
                entry.insert(0, "00")


    def obtener_hora_completa(self, horas_entry, minutos_entry):
        h = horas_entry.get().zfill(2)
        m = minutos_entry.get().zfill(2)
        return f"{h}:{m}"

    def validar_horas_colegio(self):
        # Validar horario mañana
        hora_inicio_manana_str = self.obtener_hora_completa(self.hora_inicio_manana_h, self.hora_inicio_manana_m)
        hora_fin_manana_str = self.obtener_hora_completa(self.hora_fin_manana_h, self.hora_fin_manana_m)
        
        if not self.restricciones.validar_horario_general(hora_inicio_manana_str, hora_fin_manana_str):
            messagebox.showerror("Error de Horario", "La hora de fin de la mañana debe ser posterior a la hora de inicio.")
            return False
        
        # Validar receso mañana
        if self.receso_manana_var.get():
            receso_manana_inicio_str = self.obtener_hora_completa(self.receso_manana_inicio_h, self.receso_manana_inicio_m)
            receso_manana_fin_str = self.obtener_hora_completa(self.receso_manana_fin_h, self.receso_manana_fin_m)
            if not self.restricciones.validar_receso(receso_manana_inicio_str, receso_manana_fin_str, hora_inicio_manana_str, hora_fin_manana_str):
                messagebox.showerror("Error de Receso Mañana", "El receso de la mañana debe estar dentro del horario de clases y el fin del receso debe ser posterior a su inicio.")
                return False
        
        # Validar horario tarde si está activo
        if self.turno_tarde_var.get():
            hora_inicio_tarde_str = self.obtener_hora_completa(self.hora_inicio_tarde_h, self.hora_inicio_tarde_m)
            hora_fin_tarde_str = self.obtener_hora_completa(self.hora_fin_tarde_h, self.hora_fin_tarde_m)
            
            if not self.restricciones.validar_horario_general(hora_inicio_tarde_str, hora_fin_tarde_str):
                messagebox.showerror("Error de Horario", "La hora de fin de la tarde debe ser posterior a la hora de inicio.")
                return False
            
            # Validar que no se solapen mañana y tarde
            hora_fin_manana_dt = datetime.strptime(hora_fin_manana_str, "%H:%M")
            hora_inicio_tarde_dt = datetime.strptime(hora_inicio_tarde_str, "%H:%M")
            if hora_inicio_tarde_dt <= hora_fin_manana_dt:
                messagebox.showerror("Error de Horario", "El horario de la tarde debe comenzar después del horario de la mañana.")
                return False
            
            # Validar receso tarde
            if self.receso_tarde_var.get():
                receso_tarde_inicio_str = self.obtener_hora_completa(self.receso_tarde_inicio_h, self.receso_tarde_inicio_m)
                receso_tarde_fin_str = self.obtener_hora_completa(self.receso_tarde_fin_h, self.receso_tarde_fin_m)
                if not self.restricciones.validar_receso(receso_tarde_inicio_str, receso_tarde_fin_str, hora_inicio_tarde_str, hora_fin_tarde_str):
                    messagebox.showerror("Error de Receso Tarde", "El receso de la tarde debe estar dentro del horario de clases y el fin del receso debe ser posterior a su inicio.")
                    return False
        
        return True

    # ... (los métodos crear_entradas y procesar_y_generar_horario se mantienen igual)

    def generar_horario_con_optimizacion(self, cursos_data_original):
        model = cp_model.CpModel()

        # --- CONJUNTOS Y PARÁMETROS ---
        cursos_indices = [c['id'] for c in cursos_data_original]
        dias_semana_indices = list(range(len(self.restricciones.dias_semana)))

        duracion_bloque_min = self.duracion_bloque_var.get()
        
        # Configurar horario mañana
        hora_inicio_manana_dt = datetime.strptime(self.obtener_hora_completa(self.hora_inicio_manana_h, self.hora_inicio_manana_m), "%H:%M")
        hora_fin_manana_dt = datetime.strptime(self.obtener_hora_completa(self.hora_fin_manana_h, self.hora_fin_manana_m), "%H:%M")

        # Configurar receso mañana
        receso_manana_inicio_dt = None
        receso_manana_fin_dt = None
        if self.receso_manana_var.get():
            receso_manana_inicio_dt = datetime.strptime(self.obtener_hora_completa(self.receso_manana_inicio_h, self.receso_manana_inicio_m), "%H:%M")
            receso_manana_fin_dt = datetime.strptime(self.obtener_hora_completa(self.receso_manana_fin_h, self.receso_manana_fin_m), "%H:%M")

        # Configurar horario tarde si está activo
        hora_inicio_tarde_dt = None
        hora_fin_tarde_dt = None
        receso_tarde_inicio_dt = None
        receso_tarde_fin_dt = None
        
        if self.turno_tarde_var.get():
            hora_inicio_tarde_dt = datetime.strptime(self.obtener_hora_completa(self.hora_inicio_tarde_h, self.hora_inicio_tarde_m), "%H:%M")
            hora_fin_tarde_dt = datetime.strptime(self.obtener_hora_completa(self.hora_fin_tarde_h, self.hora_fin_tarde_m), "%H:%M")
            
            if self.receso_tarde_var.get():
                receso_tarde_inicio_dt = datetime.strptime(self.obtener_hora_completa(self.receso_tarde_inicio_h, self.receso_tarde_inicio_m), "%H:%M")
                receso_tarde_fin_dt = datetime.strptime(self.obtener_hora_completa(self.receso_tarde_fin_h, self.receso_tarde_fin_m), "%H:%M")

        # Generar slots para la mañana
        daily_slots_template_manana = []
        hora_actual_dt = hora_inicio_manana_dt
        slot_idx_in_day_counter = 0
        while hora_actual_dt < hora_fin_manana_dt:
            es_receso_actual = False
            hora_fin_bloque_actual_dt = hora_actual_dt + timedelta(minutes=duracion_bloque_min)
            if receso_manana_inicio_dt and receso_manana_fin_dt:
                if hora_actual_dt < receso_manana_fin_dt and hora_fin_bloque_actual_dt > receso_manana_inicio_dt:
                    es_receso_actual = True
            daily_slots_template_manana.append({
                'dt_inicio': hora_actual_dt,
                'es_receso': es_receso_actual,
                'hora_str': hora_actual_dt.strftime("%H:%M"),
                'idx_in_day': slot_idx_in_day_counter,
                'turno': 'mañana'
            })
            hora_actual_dt += timedelta(minutes=duracion_bloque_min)
            slot_idx_in_day_counter += 1
        
        # Generar slots para la tarde si está activo
        daily_slots_template_tarde = []
        if self.turno_tarde_var.get():
            hora_actual_dt = hora_inicio_tarde_dt
            while hora_actual_dt < hora_fin_tarde_dt:
                es_receso_actual = False
                hora_fin_bloque_actual_dt = hora_actual_dt + timedelta(minutes=duracion_bloque_min)
                if receso_tarde_inicio_dt and receso_tarde_fin_dt:
                    if hora_actual_dt < receso_tarde_fin_dt and hora_fin_bloque_actual_dt > receso_tarde_inicio_dt:
                        es_receso_actual = True
                daily_slots_template_tarde.append({
                    'dt_inicio': hora_actual_dt,
                    'es_receso': es_receso_actual,
                    'hora_str': hora_actual_dt.strftime("%H:%M"),
                    'idx_in_day': slot_idx_in_day_counter,
                    'turno': 'tarde'
                })
                hora_actual_dt += timedelta(minutes=duracion_bloque_min)
                slot_idx_in_day_counter += 1
        
        # Combinar slots de mañana y tarde
        daily_slots_template = daily_slots_template_manana + daily_slots_template_tarde
        
        schedulable_slots_indices_in_template = [
            s_info['idx_in_day'] for s_info in daily_slots_template if not s_info['es_receso']
        ]
        
        if not schedulable_slots_indices_in_template:
            messagebox.showerror("Error de Configuración", "No hay bloques de clase disponibles según el horario y recesos configurados.")
            return

        H_i = {c['id']: c['horas_bloques'] for c in cursos_data_original}
        MAX_BLOQUES_DIA_CURSO = self.max_horas_diarias_var.get() * (60 // duracion_bloque_min)
        cursos_pesados_ids = [c['id'] for c in cursos_data_original if c['nombre'] in self.restricciones.cursos_pesados_nombres]

        # --- VARIABLES DE DECISIÓN x_ijt ---
        x = {}
        for i in cursos_indices:
            for j in dias_semana_indices:
                for t_slot_idx in schedulable_slots_indices_in_template:
                    x[i, j, t_slot_idx] = model.NewBoolVar(f'x_{i}_{j}_{t_slot_idx}')

        # --- RESTRICCIONES ---
        # 1) Total horas (bloques) por curso
        for i in cursos_indices:
            model.Add(sum(x[i, j, t_slot_idx] 
                      for j in dias_semana_indices 
                      for t_slot_idx in schedulable_slots_indices_in_template) == H_i[i])

        # 2) Un solo curso por bloque horario
        for j in dias_semana_indices:
            for t_slot_idx in schedulable_slots_indices_in_template:
                model.Add(sum(x[i, j, t_slot_idx] for i in cursos_indices) <= 1)
        
        # 3) Máximo MAX_BLOQUES_DIA_CURSO por curso y día
        for i in cursos_indices:
            for j in dias_semana_indices:
                model.Add(sum(x[i, j, t_slot_idx] for t_slot_idx in schedulable_slots_indices_in_template) <= MAX_BLOQUES_DIA_CURSO)

        # 4) Cursos pesados (M) no coinciden en el mismo día
        if self.restricciones.evitar_mismo_dia_cursos_pesados and len(cursos_pesados_ids) >= 2:
            b = {} 
            for i_p in cursos_pesados_ids:
                for j in dias_semana_indices:
                    b[i_p, j] = model.NewBoolVar(f'b_{i_p}_{j}')
                    curso_dia_vars = [x[i_p, j, t_slot_idx] for t_slot_idx in schedulable_slots_indices_in_template]
                    if curso_dia_vars:
                        model.AddMaxEquality(b[i_p, j], curso_dia_vars)
                    else:
                        model.Add(b[i_p,j] == 0)

            for j in dias_semana_indices:
                for idx1 in range(len(cursos_pesados_ids)):
                    for idx2 in range(idx1 + 1, len(cursos_pesados_ids)):
                        i1 = cursos_pesados_ids[idx1]
                        i2 = cursos_pesados_ids[idx2]
                        if (i1,j) in b and (i2,j) in b:
                            model.Add(b[i1, j] + b[i2, j] <= 1)
        
        # --- FUNCIÓN OBJETIVO ---
        # Parte 1: Balance (minimizar suma de cuadrados de bloques diarios por curso)
        x_ij_val = {} 
        for i in cursos_indices:
            for j in dias_semana_indices:
                x_ij_val[i,j] = model.NewIntVar(0, MAX_BLOQUES_DIA_CURSO, f'x_ij_val_{i}_{j}')
                model.Add(x_ij_val[i,j] == sum(x[i, j, t_slot_idx] 
                                         for t_slot_idx in schedulable_slots_indices_in_template))
        
        obj_terms_balance = []
        for i in cursos_indices:
            for j in dias_semana_indices:
                sq_var = model.NewIntVar(0, MAX_BLOQUES_DIA_CURSO * MAX_BLOQUES_DIA_CURSO, f'x_ij_sq_{i}_{j}')
                model.AddMultiplicationEquality(sq_var, x_ij_val[i,j], x_ij_val[i,j])
                obj_terms_balance.append(sq_var)
        
        objetivo_balance_total = sum(obj_terms_balance)

        # Parte 2: Priorizar cursos pesados en slots tempranos
        obj_terms_prioridad_temprana = []
        WEIGHT_PRIORIDAD_TEMPRANA = 1

        for i_curso_id in cursos_pesados_ids:
            for j_dia_idx in dias_semana_indices:
                for t_slot_real_idx in schedulable_slots_indices_in_template:
                    if (i_curso_id, j_dia_idx, t_slot_real_idx) in x:
                        # Penalizar más los bloques de la tarde (si existen)
                        slot_info = daily_slots_template[t_slot_real_idx]
                        peso_turno = 2 if slot_info['turno'] == 'tarde' else 1
                        obj_terms_prioridad_temprana.append(
                            peso_turno * t_slot_real_idx * x[i_curso_id, j_dia_idx, t_slot_real_idx]
                        )
        
        objetivo_prioridad_total = sum(obj_terms_prioridad_temprana)

        # Combinar los objetivos
        if obj_terms_prioridad_temprana:
            model.Minimize(objetivo_balance_total + WEIGHT_PRIORIDAD_TEMPRANA * objetivo_prioridad_total)
        else:
            model.Minimize(objetivo_balance_total)

        # --- RESOLVER EL MODELO ---
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0
        status = solver.Solve(model)

        # --- PROCESAR Y MOSTRAR RESULTADOS ---
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            if status == cp_model.OPTIMAL:
                print(f"Solución óptima encontrada. Objetivo balance: {solver.Value(objetivo_balance_total)}, Objetivo prioridad: {solver.Value(objetivo_prioridad_total) if obj_terms_prioridad_temprana else 0 }")
            else:
                print(f"Solución factible encontrada. Objetivo balance: {solver.Value(objetivo_balance_total)}, Objetivo prioridad: {solver.Value(objetivo_prioridad_total) if obj_terms_prioridad_temprana else 0 }")

            horario_final_pdf = {}
            for dia_idx_enum, dia_str in enumerate(self.restricciones.dias_semana):
                horario_final_pdf[dia_str] = {}
                for slot_info in daily_slots_template:
                    hora_str = slot_info['hora_str']
                    if slot_info['es_receso']:
                        horario_final_pdf[dia_str][hora_str] = "RECESO"
                    else:
                        horario_final_pdf[dia_str][hora_str] = "Libre"

            for i_curso_idx in cursos_indices:
                curso_nombre_actual = next(c['nombre'] for c in cursos_data_original if c['id'] == i_curso_idx)
                for j_dia_idx, dia_str_actual in enumerate(self.restricciones.dias_semana):
                    for t_slot_real_idx in schedulable_slots_indices_in_template:
                        if (i_curso_idx, j_dia_idx, t_slot_real_idx) in x and \
                           solver.Value(x[i_curso_idx, j_dia_idx, t_slot_real_idx]) == 1:
                            hora_str_asignada = daily_slots_template[t_slot_real_idx]['hora_str']
                            if horario_final_pdf[dia_str_actual][hora_str_asignada] != "Libre":
                                print(f"ADVERTENCIA DE SOLAPAMIENTO DEL SOLVER: Slot {dia_str_actual} {hora_str_asignada} que ya tenía {horario_final_pdf[dia_str_actual][hora_str_asignada]} se intenta asignar a {curso_nombre_actual}")
                            horario_final_pdf[dia_str_actual][hora_str_asignada] = curso_nombre_actual
            
            # Generar PDF
            pdf = FPDF(orientation='L')
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(0, 10, "Horario Escolar Optimizado (OR-Tools)", ln=True, align="C")
            pdf.set_font("Arial", size=10)
            pdf.ln(5)

            # Información del horario
            hora_inicio_manana_str = self.obtener_hora_completa(self.hora_inicio_manana_h, self.hora_inicio_manana_m)
            hora_fin_manana_str = self.obtener_hora_completa(self.hora_fin_manana_h, self.hora_fin_manana_m)
            pdf.cell(0, 6, f"Horario Mañana: {hora_inicio_manana_str} - {hora_fin_manana_str}", ln=True)
            
            if self.turno_tarde_var.get():
                hora_inicio_tarde_str = self.obtener_hora_completa(self.hora_inicio_tarde_h, self.hora_inicio_tarde_m)
                hora_fin_tarde_str = self.obtener_hora_completa(self.hora_fin_tarde_h, self.hora_fin_tarde_m)
                pdf.cell(0, 6, f"Horario Tarde: {hora_inicio_tarde_str} - {hora_fin_tarde_str}", ln=True)
            
            pdf.cell(0, 6, f"Total horas semanales colegio: {self.horas_semanales_var.get()}", ln=True)
            pdf.cell(0, 6, f"Duración bloque: {self.duracion_bloque_var.get()} min.", ln=True)
            
            if self.receso_manana_var.get():
                receso_manana_i = self.obtener_hora_completa(self.receso_manana_inicio_h, self.receso_manana_inicio_m)
                receso_manana_f = self.obtener_hora_completa(self.receso_manana_fin_h, self.receso_manana_fin_m)
                pdf.cell(0, 6, f"Receso Mañana: {receso_manana_i} - {receso_manana_f}", ln=True)
            
            if self.turno_tarde_var.get() and self.receso_tarde_var.get():
                receso_tarde_i = self.obtener_hora_completa(self.receso_tarde_inicio_h, self.receso_tarde_inicio_m)
                receso_tarde_f = self.obtener_hora_completa(self.receso_tarde_fin_h, self.receso_tarde_fin_m)
                pdf.cell(0, 6, f"Receso Tarde: {receso_tarde_i} - {receso_tarde_f}", ln=True)
            
            pdf.ln(5)

            # Tabla del horario
            pdf.set_font("Arial", "B", 9)
            pdf.set_fill_color(200, 220, 255)
            ancho_col_hora = 25
            ancho_col_dia = (pdf.w - 30 - ancho_col_hora) / len(self.restricciones.dias_semana)

            pdf.cell(ancho_col_hora, 8, "Hora", 1, 0, 'C', True)
            for dia_str_pdf in self.restricciones.dias_semana:
                pdf.cell(ancho_col_dia, 8, dia_str_pdf, 1, 0, 'C', True)
            pdf.ln()

            pdf.set_font("Arial", size=8)
            pdf.set_fill_color(255, 255, 255)
            
            for slot_info_template in daily_slots_template:
                hora_actual_str_pdf = slot_info_template['hora_str']
                pdf.cell(ancho_col_hora, 7, hora_actual_str_pdf, 1, 0, 'C')
                for dia_str_pdf_col in self.restricciones.dias_semana:
                    contenido_celda = horario_final_pdf[dia_str_pdf_col].get(hora_actual_str_pdf, "Error")
                    fill_celda = False
                    if contenido_celda == "RECESO":
                        pdf.set_fill_color(220, 220, 220)
                        fill_celda = True
                    pdf.cell(ancho_col_dia, 7, str(contenido_celda), 1, 0, 'C', fill_celda)
                    if fill_celda: 
                        pdf.set_fill_color(255,255,255)
                pdf.ln()

            pdf_filename = "Horario_Optimizado_ORTools.pdf"
            try:
                pdf.output(pdf_filename)
                messagebox.showinfo("Éxito", f"Horario generado exitosamente en '{pdf_filename}'")
            except Exception as e_pdf:
                messagebox.showerror("Error al Guardar PDF", f"No se pudo guardar el PDF: {e_pdf}")

        elif status == cp_model.INFEASIBLE:
            messagebox.showerror("Sin Solución", "No se encontró una solución factible con las restricciones dadas. Pruebe:\n"
                             "- Aumentar horas semanales del colegio.\n"
                             "- Reducir horas totales de los cursos.\n"
                             "- Aumentar el máximo de horas diarias por curso.\n"
                             "- Ajustar los horarios de receso o la duración de los bloques.\n"
                             "- Reducir el número de 'cursos pesados' o desactivar su restricción de no coincidencia.")
        elif status == cp_model.MODEL_INVALID:
            messagebox.showerror("Error de Modelo", "El modelo de optimización es inválido. Revise los parámetros y restricciones. Detalles en consola.")
        else:
            messagebox.showwarning("Resultado Incierto", f"El solver terminó con estado: {status}. No se garantiza una solución óptima o factible.")

if __name__ == "__main__":
    app = OptimizadorHorarios()
    app.mainloop()