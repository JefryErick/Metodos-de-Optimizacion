import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import random
from collections import Counter, defaultdict
import math

# -------------------- CONSTANTES --------------------
TIPOS_CARAMELOS = ['limon', 'pera', 'bola']
CARAMELITOS_POR_JUGADOR = 2
JUGADORES_POR_GRUPO = 3
REQUISITO_CHUPETIN = {'limon': 2, 'pera': 2, 'bola': 2}

# Colores para cada tipo de caramelo
COLORES_CARAMELOS = {
    'limon': '#FFD700',  # Dorado
    'pera': '#90EE90',   # Verde claro
    'bola': '#FF69B4'    # Rosa
}

# -------------------- LÓGICA MEJORADA --------------------
class Jugador:
    def __init__(self, id):
        self.id = id
        self.caramelos = random.choices(TIPOS_CARAMELOS, k=CARAMELITOS_POR_JUGADOR)
        self.salvado = False
        self.chupetines = 0
        self.x = 0
        self.y = 0
        self.canvas_id = None
        self.animating = False

    def __repr__(self):
        return f"Jugador {self.id} ({self.caramelos})"

class Grupo:
    def __init__(self, jugadores):
        self.jugadores = jugadores
        self.chupetines = 0

    def caramelos_totales(self):
        contador = Counter()
        for jugador in self.jugadores:
            contador.update(jugador.caramelos)
        return contador

# -------------------- INTERFAZ VISUAL MEJORADA --------------------
class CanvasJugadores(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg='#2C3E50', **kwargs)
        self.jugadores = []
        self.animaciones_activas = []
        
    def dibujar_jugadores(self, jugadores, grupos):
        self.delete("all")
        self.jugadores = jugadores
        
        # Organizar jugadores en grupos visualmente
        margen = 50
        ancho_grupo = 180
        alto_grupo = 120
        
        for i, grupo in enumerate(grupos):
            # Calcular posición del grupo
            cols = 3  # máximo 3 grupos por fila
            fila = i // cols
            col = i % cols
            
            grupo_x = margen + col * (ancho_grupo + 30)
            grupo_y = margen + fila * (alto_grupo + 40)
            
            # Dibujar marco del grupo
            self.create_rectangle(grupo_x - 10, grupo_y - 10, 
                                grupo_x + ancho_grupo, grupo_y + alto_grupo,
                                outline='#34495E', width=2, fill='#34495E')
            
            # Título del grupo
            self.create_text(grupo_x + ancho_grupo//2, grupo_y - 25,
                           text=f"Grupo {i+1}", fill='white', font=('Arial', 12, 'bold'))
            
            # Dibujar jugadores del grupo
            for j, jugador in enumerate(grupo.jugadores):
                jugador_x = grupo_x + 20 + (j % 2) * 80
                jugador_y = grupo_y + 20 + (j // 2) * 50
                
                jugador.x = jugador_x
                jugador.y = jugador_y
                
                self.dibujar_jugador(jugador)
    
    def dibujar_jugador(self, jugador):
        x, y = jugador.x, jugador.y
        
        # Color del jugador según estado
        color_jugador = '#27AE60' if jugador.salvado else '#E74C3C'
        
        # Círculo del jugador
        radio = 20
        jugador.canvas_id = self.create_oval(x - radio, y - radio, 
                                           x + radio, y + radio,
                                           fill=color_jugador, outline='white', width=2)
        
        # Número del jugador
        self.create_text(x, y - 5, text=str(jugador.id), 
                        fill='white', font=('Arial', 10, 'bold'))
        
        # Chupetines
        if jugador.chupetines > 0:
            self.create_text(x, y + 8, text=f"🍭{jugador.chupetines}", 
                           fill='white', font=('Arial', 8))
        
        # Caramelos del jugador
        for i, caramelo in enumerate(jugador.caramelos):
            caramelo_x = x - 15 + i * 8
            caramelo_y = y + 30
            
            self.create_oval(caramelo_x - 4, caramelo_y - 4,
                           caramelo_x + 4, caramelo_y + 4,
                           fill=COLORES_CARAMELOS[caramelo],
                           outline='black', width=1)
    
    def animar_canje(self, jugador):
        """Anima el efecto de canje"""
        if jugador.canvas_id:
            # Efecto de brillo
            for i in range(5):
                self.after(i * 100, lambda i=i: self.efecto_brillo(jugador))
    
    def efecto_brillo(self, jugador):
        if jugador.canvas_id:
            # Cambiar color temporalmente
            self.itemconfig(jugador.canvas_id, fill='#F39C12')
            self.after(100, lambda: self.itemconfig(jugador.canvas_id, fill='#27AE60'))

class BarraCaramelos(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg='#34495E')
        self.barras = {}
        self.crear_barras()
    
    def crear_barras(self):
        title = tk.Label(self, text="Caramelos en Bolsa Global", 
                        bg='#34495E', fg='white', font=('Arial', 14, 'bold'))
        title.pack(pady=10)
        
        for tipo in TIPOS_CARAMELOS:
            frame = tk.Frame(self, bg='#34495E')
            frame.pack(fill='x', padx=20, pady=5)
            
            # Etiqueta del tipo
            label = tk.Label(frame, text=tipo.capitalize(), 
                           bg='#34495E', fg='white', font=('Arial', 12))
            label.pack(side='left')
            
            # Canvas para la barra
            canvas = tk.Canvas(frame, height=25, bg='#2C3E50', highlightthickness=0)
            canvas.pack(side='right', fill='x', expand=True, padx=(10, 0))
            
            self.barras[tipo] = canvas
    
    def actualizar(self, bolsa_global):
        max_valor = max(bolsa_global.values()) if bolsa_global.values() else 1
        
        for tipo, canvas in self.barras.items():
            canvas.delete("all")
            cantidad = bolsa_global.get(tipo, 0)
            
            # Calcular ancho de la barra
            ancho_total = canvas.winfo_width()
            if ancho_total <= 1:  # Canvas aún no renderizado
                ancho_total = 200
            
            ancho_barra = int((cantidad / max(max_valor, 1)) * ancho_total * 0.8)
            
            # Dibujar barra
            color = COLORES_CARAMELOS[tipo]
            canvas.create_rectangle(5, 5, 5 + ancho_barra, 20,
                                  fill=color, outline='white')
            
            # Mostrar cantidad
            canvas.create_text(ancho_total - 20, 12, text=str(cantidad),
                             fill='white', font=('Arial', 10, 'bold'))

class PanelControl(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg='#34495E')
        self.app = app
        self.crear_controles()
    
    def crear_controles(self):
        # Título
        title = tk.Label(self, text="Panel de Control", 
                        bg='#34495E', fg='white', font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Frame para entrada de jugadores
        frame_jugadores = tk.Frame(self, bg='#34495E')
        frame_jugadores.pack(pady=10)
        
        tk.Label(frame_jugadores, text="Jugadores:", 
                bg='#34495E', fg='white', font=('Arial', 12)).pack(side='left')
        
        self.entrada_jugadores = tk.Entry(frame_jugadores, width=10, font=('Arial', 12))
        self.entrada_jugadores.pack(side='left', padx=5)
        self.entrada_jugadores.insert(0, "6")
        
        # Botones principales
        btn_iniciar = tk.Button(self, text="🎮 Iniciar Juego", 
                               command=self.app.iniciar_juego,
                               bg='#3498DB', fg='white', font=('Arial', 12, 'bold'),
                               relief='flat', padx=20, pady=5)
        btn_iniciar.pack(pady=5, fill='x', padx=20)
        
        btn_canje = tk.Button(self, text="✨ Realizar Canje", 
                             command=self.app.optimizar_paso_a_paso,
                             bg='#E67E22', fg='white', font=('Arial', 12, 'bold'),
                             relief='flat', padx=20, pady=5)
        btn_canje.pack(pady=5, fill='x', padx=20)
        
        btn_auto = tk.Button(self, text="🤖 Modo Automático", 
                            command=self.app.modo_automatico,
                            bg='#9B59B6', fg='white', font=('Arial', 12, 'bold'),
                            relief='flat', padx=20, pady=5)
        btn_auto.pack(pady=5, fill='x', padx=20)
        
        btn_solucion = tk.Button(self, text="💡 Mostrar Solución", 
                               command=self.app.mostrar_solucion_optima,
                               bg='#1ABC9C', fg='white', font=('Arial', 12, 'bold'),
                               relief='flat', padx=20, pady=5)
        btn_solucion.pack(pady=5, fill='x', padx=20)
        
        # Estadísticas
        self.frame_stats = tk.Frame(self, bg='#2C3E50')
        self.frame_stats.pack(fill='both', expand=True, padx=20, pady=20)
        
        tk.Label(self.frame_stats, text="Estadísticas", 
                bg='#2C3E50', fg='white', font=('Arial', 14, 'bold')).pack(pady=5)
        
        self.label_stats = tk.Label(self.frame_stats, text="", 
                                   bg='#2C3E50', fg='white', font=('Arial', 10),
                                   justify='left')
        self.label_stats.pack(pady=5)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🍭 Juego de Optimización de Caramelos - Versión Visual")
        self.geometry("1200x800")
        self.configure(bg='#2C3E50')
        
        # Datos del juego
        self.jugadores = []
        self.grupos = []
        self.bolsa_global = Counter()
        self.historial = []
        self.modo_auto_activo = False
        
        self.crear_interfaz()
        
    def crear_interfaz(self):
        # Panel principal dividido
        paned_main = ttk.PanedWindow(self, orient='horizontal')
        paned_main.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Panel izquierdo - Visualización
        frame_izq = tk.Frame(paned_main, bg='#34495E')
        paned_main.add(frame_izq, weight=3)
        
        # Canvas para jugadores
        self.canvas_jugadores = CanvasJugadores(frame_izq, height=400)
        self.canvas_jugadores.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Barra de caramelos
        self.barra_caramelos = BarraCaramelos(frame_izq)
        self.barra_caramelos.pack(fill='x', padx=10, pady=5)
        
        # Panel derecho - Control
        self.panel_control = PanelControl(paned_main, self)
        paned_main.add(self.panel_control, weight=1)
    
    def iniciar_juego(self):
        try:
            entrada = self.panel_control.entrada_jugadores.get()
            if not entrada.isdigit():
                messagebox.showerror("Error", "Ingrese un número válido de jugadores.")
                return
            
            n = int(entrada)
            if n % 3 != 0:
                messagebox.showerror("Error", "El número debe ser múltiplo de 3.")
                return
            
            if n < 3 or n > 30:
                messagebox.showerror("Error", "Número de jugadores debe estar entre 3 y 30.")
                return
            
            # Reiniciar juego
            self.modo_auto_activo = False
            self.jugadores = [Jugador(i+1) for i in range(n)]
            self.grupos = [Grupo(self.jugadores[i:i+3]) for i in range(0, n, 3)]
            self.bolsa_global = Counter()
            
            for jugador in self.jugadores:
                self.bolsa_global.update(jugador.caramelos)
            
            self.historial = []
            
            # Actualizar visualización
            self.canvas_jugadores.dibujar_jugadores(self.jugadores, self.grupos)
            self.actualizar_interfaz()
            
            messagebox.showinfo("Éxito", f"Juego iniciado con {n} jugadores!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar el juego: {str(e)}")
    
    def puede_canjear(self):
        return all(self.bolsa_global[tipo] >= REQUISITO_CHUPETIN[tipo] for tipo in TIPOS_CARAMELOS)
    
    def elegir_mejores_caramelos(self):
        # Estrategia mejorada para elegir caramelos
        faltantes = Counter()
        for tipo in TIPOS_CARAMELOS:
            faltantes[tipo] = max(0, REQUISITO_CHUPETIN[tipo] - self.bolsa_global[tipo])
        
        if sum(faltantes.values()) > 0:
            # Elegir los que más faltan
            candidatos = faltantes.most_common()
            return [tipo for tipo, _ in candidatos[:2]]
        else:
            # Si no faltan, equilibrar
            minimo = min(self.bolsa_global.values())
            candidatos = [tipo for tipo, cant in self.bolsa_global.items() if cant == minimo]
            return candidatos[:2] if len(candidatos) >= 2 else candidatos + [random.choice(TIPOS_CARAMELOS)]
    
    def optimizar_paso_a_paso(self):
        if not self.jugadores:
            messagebox.showerror("Error", "Primero inicie el juego.")
            return
        
        if not self.puede_canjear():
            messagebox.showwarning("Advertencia", "No hay suficientes caramelos para canjear.")
            return
        
        # Realizar canje
        for tipo in TIPOS_CARAMELOS:
            self.bolsa_global[tipo] -= REQUISITO_CHUPETIN[tipo]
        
        # Encontrar jugador no salvado
        jugadores_no_salvados = [j for j in self.jugadores if not j.salvado]
        if not jugadores_no_salvados:
            messagebox.showinfo("¡Completado!", "¡Todos los jugadores ya están salvados!")
            return
        
        jugador = jugadores_no_salvados[0]
        jugador.salvado = True
        jugador.chupetines += 1
        
        # Elegir nuevos caramelos estratégicamente
        nuevos = self.elegir_mejores_caramelos()
        jugador.caramelos.extend(nuevos)
        self.bolsa_global.update(nuevos)
        
        # Registrar acción
        accion = f"Jugador {jugador.id} salvado. Nuevos caramelos: {nuevos}"
        self.historial.append(accion)
        
        # Animación
        self.canvas_jugadores.animar_canje(jugador)
        
        # Actualizar interfaz
        self.after(500, self.actualizar_interfaz)  # Delay para la animación
        
        # Verificar si todos están salvados
        if all(j.salvado for j in self.jugadores):
            self.after(1000, lambda: messagebox.showinfo("¡Victoria!", "¡Todos los jugadores han sido salvados!"))
    
    def modo_automatico(self):
        if not self.jugadores:
            messagebox.showerror("Error", "Primero inicie el juego.")
            return
        
        if self.modo_auto_activo:
            self.modo_auto_activo = False
            return
        
        self.modo_auto_activo = True
        self.ejecutar_modo_auto()
    
    def ejecutar_modo_auto(self):
        if not self.modo_auto_activo:
            return
        
        if self.puede_canjear() and any(not j.salvado for j in self.jugadores):
            self.optimizar_paso_a_paso()
            self.after(2000, self.ejecutar_modo_auto)  # Continuar en 2 segundos
        else:
            self.modo_auto_activo = False
            if all(j.salvado for j in self.jugadores):
                messagebox.showinfo("¡Completado!", "¡Modo automático completado exitosamente!")
            else:
                messagebox.showwarning("Detenido", "Modo automático detenido - no hay suficientes caramelos.")
    
    def actualizar_interfaz(self):
        # Actualizar visualización de jugadores
        self.canvas_jugadores.dibujar_jugadores(self.jugadores, self.grupos)
        
        # Actualizar barra de caramelos
        self.barra_caramelos.actualizar(self.bolsa_global)
        
        # Actualizar estadísticas
        salvados = sum(1 for j in self.jugadores if j.salvado)
        total = len(self.jugadores)
        progreso = (salvados / total) * 100 if total > 0 else 0
        
        stats_text = f"""Jugadores: {total}
Salvados: {salvados} ({progreso:.1f}%)
Por salvar: {total - salvados}

Posibles canjes: {self.contar_canjes_posibles()}

Último canje: {self.historial[-1] if self.historial else 'Ninguno'}"""
        
        self.panel_control.label_stats.config(text=stats_text)
    
    def contar_canjes_posibles(self):
        canjes = 0
        bolsa_temp = self.bolsa_global.copy()
        
        while all(bolsa_temp[tipo] >= REQUISITO_CHUPETIN[tipo] for tipo in TIPOS_CARAMELOS):
            canjes += 1
            for tipo in TIPOS_CARAMELOS:
                bolsa_temp[tipo] -= REQUISITO_CHUPETIN[tipo]
            # Simular caramelos nuevos
            for tipo in TIPOS_CARAMELOS[:2]:  # Simplificación
                bolsa_temp[tipo] += 1
        
        return canjes
    
    def mostrar_solucion_optima(self):
        if not self.jugadores:
            messagebox.showerror("Error", "Primero inicie el juego.")
            return
        
        # Ventana de solución mejorada
        ventana = tk.Toplevel(self)
        ventana.title("💡 Solución Óptima")
        ventana.geometry("700x500")
        ventana.configure(bg='#2C3E50')
        
        # Frame principal con scroll
        main_frame = tk.Frame(ventana, bg='#2C3E50')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Título
        titulo = tk.Label(main_frame, text="🧠 Análisis de Solución Óptima", 
                         bg='#2C3E50', fg='white', font=('Arial', 18, 'bold'))
        titulo.pack(pady=10)
        
        # Análisis actual
        frame_analisis = tk.Frame(main_frame, bg='#34495E', relief='raised', bd=2)
        frame_analisis.pack(fill='x', pady=10)
        
        tk.Label(frame_analisis, text="📊 Estado Actual", 
                bg='#34495E', fg='white', font=('Arial', 14, 'bold')).pack(pady=5)
        
        total_jugadores = len(self.jugadores)
        salvados = sum(1 for j in self.jugadores if j.salvado)
        canjes_posibles = self.contar_canjes_posibles()
        
        info_actual = f"""Jugadores totales: {total_jugadores}
Jugadores salvados: {salvados}
Jugadores por salvar: {total_jugadores - salvados}
Canjes posibles con caramelos actuales: {canjes_posibles}

Caramelos en bolsa:
• Limón: {self.bolsa_global['limon']} 🍋
• Pera: {self.bolsa_global['pera']} 🍐  
• Bola: {self.bolsa_global['bola']} 🔴"""
        
        tk.Label(frame_analisis, text=info_actual, 
                bg='#34495E', fg='white', font=('Arial', 11), justify='left').pack(pady=10)
        
        # Estrategia recomendada
        frame_estrategia = tk.Frame(main_frame, bg='#27AE60', relief='raised', bd=2)
        frame_estrategia.pack(fill='x', pady=10)
        
        tk.Label(frame_estrategia, text="🎯 Estrategia Recomendada", 
                bg='#27AE60', fg='white', font=('Arial', 14, 'bold')).pack(pady=5)
        
        estrategia = """1. 🔄 Realiza canjes cuando sea posible (2 de cada tipo → 1 chupetín)
2. 🎯 Al elegir caramelos nuevos, prioriza los que menos tienes
3. 📈 Cada canje te da 2 caramelos nuevos, optimiza para futuros canjes
4. 🤖 Usa el modo automático para ver la solución completa
5. ⚡ La clave es mantener balance entre los 3 tipos de caramelos"""
        
        tk.Label(frame_estrategia, text=estrategia, 
                bg='#27AE60', fg='white', font=('Arial', 11), justify='left').pack(pady=10)
        
        # Botón para cerrar
        tk.Button(main_frame, text="✅ Entendido", command=ventana.destroy,
                 bg='#3498DB', fg='white', font=('Arial', 12, 'bold'),
                 relief='flat', padx=30, pady=10).pack(pady=20)

if __name__ == "__main__":
    app = App()
    app.mainloop()


"""
Método de Optimización Utilizado
El código implementa una estrategia greedy (voraz) para resolver el problema de optimización de caramelos, con las siguientes características:

Enfoque paso a paso: En cada iteración, el algoritmo toma la decisión localmente óptima (canjear caramelos cuando es posible) sin considerar el impacto global a largo plazo.

Función de selección: El método elegir_mejores_caramelos() implementa la lógica greedy:

Primero prioriza los caramelos que faltan para completar el próximo canje

Si no faltan, elige los caramelos menos abundantes para mantener un balance

Objetivo inmediato: Salvar jugadores uno por uno sin una planificación estratégica a largo plazo.
"""