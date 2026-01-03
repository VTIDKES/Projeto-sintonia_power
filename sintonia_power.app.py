import numpy as np
import matplotlib.pyplot as plt

class PowerSystemSimulator:
    def __init__(self):
        # Parâmetros do Gerador
        self.V0_mag = 1.05  # Magnitude da tensão (p.u.)
        self.V0_angle = 0   # Ângulo em graus
        
        # Impedâncias da Linha de Distribuição (0-1)
        self.R01 = 0.02  # Resistência (p.u.)
        self.X01 = 0.06  # Reatância (p.u.)
        
        # Impedâncias da Linha de Transmissão (1-2)
        self.R12 = 0.05  # Resistência (p.u.)
        self.X12 = 0.15  # Reatância (p.u.)
        
        # Carga no barramento 1
        self.P_load = 0.8  # Potência ativa (p.u.)
        self.Q_load = 0.4  # Potência reativa (p.u.)
        
        # Barramento Infinito
        self.V2_mag = 1.0   # Magnitude da tensão (p.u.)
        self.V2_angle = 0   # Ângulo em graus
        
    def calculate_power_flow(self, max_iter=100, tolerance=1e-6):
        """
        Calcula o fluxo de potência usando método iterativo
        """
        # Converter ângulos para radianos
        V0_angle_rad = np.deg2rad(self.V0_angle)
        V2_angle_rad = np.deg2rad(self.V2_angle)
        
        # Tensões complexas
        V0 = self.V0_mag * np.exp(1j * V0_angle_rad)
        V2 = self.V2_mag * np.exp(1j * V2_angle_rad)
        
        # Impedâncias complexas
        Z01 = self.R01 + 1j * self.X01
        Z12 = self.R12 + 1j * self.X12
        
        # Estimativa inicial para V1
        V1 = 1.0 + 0j
        
        # Método iterativo de Gauss-Seidel
        for iteration in range(max_iter):
            V1_old = V1
            
            # Corrente da carga (S = V * I*, então I = (S/V)*)
            S_load = self.P_load + 1j * self.Q_load
            I_load = np.conj(S_load / V1)
            
            # Corrente na linha 1-2
            I12 = (V1 - V2) / Z12
            
            # Corrente total no nó 1
            I1 = I_load + I12
            
            # Atualizar V1 usando a equação do nó
            V1 = V0 - I1 * Z01
            
            # Verificar convergência
            if abs(V1 - V1_old) < tolerance:
                print(f"Convergiu em {iteration + 1} iterações")
                break
        
        # Calcular correntes finais
        I_load = np.conj(S_load / V1)
        I12 = (V1 - V2) / Z12
        I1 = I_load + I12
        I01 = I1  # Corrente na linha 0-1
        
        # Calcular potências
        S_gen = V0 * np.conj(I01)  # Potência gerada
        S_load_calc = V1
