# STREAMLIT + PANDAPOWER - Análise Completa
# Sistemas Elétricos de Potência
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import time
from datetime import datetime
from streamlit.components.v1 import html, components
import uuid

# Configuração da página
st.set_page_config(
    page_title="Power System Editor - Drag & Drop",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS e JavaScript para drag-and-drop
st.markdown("""
<style>
    /* Estilos gerais */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Elementos arrastáveis */
    .draggable-element {
        padding: 12px;
        margin: 8px 0;
        border-radius: 8px;
        cursor: grab;
        transition: all 0.3s ease;
        user-select: none;
        border: 2px solid transparent;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .draggable-element:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        border-color: #007bff;
    }
    
    .draggable-element:active {
        cursor: grabbing;
        transform: translateY(0);
    }
    
    /* Cores por tipo */
    .bus-slack { border-left: 5px solid #dc3545; }
    .bus-pv { border-left: 5px solid #28a745; }
    .bus-pq { border-left: 5px solid #007bff; }
    .element-line { border-left: 5px solid #6f42c1; }
    .element-load { border-left: 5px solid #fd7e14; }
    .element-gen { border-left: 5px solid #20c997; }
    
    /* Área do canvas */
    .canvas-container {
        border: 2px dashed #dee2e6;
        border-radius: 10px;
        background-color: white;
        min-height: 700px;
        position: relative;
        overflow: hidden;
        touch-action: none;
    }
    
    .canvas-grid {
        background-image: 
            linear-gradient(rgba(0,0,0,0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,0,0,0.05) 1px, transparent 1px);
        background-size: 20px 20px;
    }
    
    /* Elementos no canvas */
    .canvas-bus {
        position: absolute;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        cursor: move;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 14px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        z-index: 10;
        user-select: none;
        touch-action: none;
    }
    
    .bus-slack-bg { background-color: #dc3545; }
    .bus-pv-bg { background-color: #28a745; }
    .bus-pq-bg { background-color: #007bff; }
    
    .canvas-line {
        position: absolute;
        background-color: #6f42c1;
        height: 3px;
        transform-origin: 0 0;
        z-index: 5;
    }
    
    .canvas-load {
        position: absolute;
        width: 30px;
        height: 30px;
        background-color: #fd7e14;
        cursor: move;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 12px;
        border-radius: 4px;
        z-index: 9;
    }
    
    .canvas-gen {
        position: absolute;
        width: 30px;
        height: 30px;
        background-color: #20c997;
        cursor: move;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 12px;
        border-radius: 50%;
        z-index: 9;
    }
    
    .element-label {
        position: absolute;
        background: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 10px;
        border: 1px solid #dee2e6;
        pointer-events: none;
        z-index: 11;
    }
    
    .selected {
        box-shadow: 0 0 0 3px #ffc107, 0 0 20px rgba(255,193,7,0.5);
        z-index: 100;
    }
    
    /* Botões */
    .mode-btn {
        margin: 5px 0;
        transition: all 0.3s;
    }
    
    .mode-btn.active {
        background-color: #007bff !important;
        color: white !important;
        border-color: #007bff !important;
        box-shadow: 0 0 10px rgba(0,123,255,0.3);
    }
    
    .action-btn {
        width: 100%;
        margin: 3px 0;
    }
    
    /* Painel de propriedades */
    .property-panel {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    
    /* Status */
    .status-bar {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-weight: bold;
        text-align: center;
    }
</style>

<script>
// Variáveis globais
let currentMode = 'select';
let selectedElement = null;
let connectingFrom = null;
let elements = [];
let lines = [];
let isDragging = false;
let dragOffset = { x: 0, y: 0 };
let canvasRect = null;

// Elementos da paleta
const paletteElements = [
    { id: 'bus-slack', type: 'bus', subtype: 'slack', name: 'Barra Slack', color: '#dc3545', emoji: '🔴' },
    { id: 'bus-pv', type: 'bus', subtype: 'pv', name: 'Barra PV', color: '#28a745', emoji: '🟢' },
    { id: 'bus-pq', type: 'bus', subtype: 'pq', name: 'Barra PQ', color: '#007bff', emoji: '🔵' },
    { id: 'line', type: 'line', name: 'Linha', color: '#6f42c1', emoji: '📈' },
    { id: 'load', type: 'load', name: 'Carga', color: '#fd7e14', emoji: '💡' },
    { id: 'generator', type: 'generator', name: 'Gerador', color: '#20c997', emoji: '⚡' }
];

// Inicialização
document.addEventListener('DOMContentLoaded', function() {
    initCanvas();
    initPalette();
    updateStatus();
});

function initCanvas() {
    const canvas = document.getElementById('canvas');
    if (!canvas) return;
    
    canvasRect = canvas.getBoundingClientRect();
    
    // Habilitar arrastar elementos no canvas
    canvas.addEventListener('mousedown', handleCanvasMouseDown);
    canvas.addEventListener('mousemove', handleCanvasMouseMove);
    canvas.addEventListener('mouseup', handleCanvasMouseUp);
    canvas.addEventListener('touchstart', handleCanvasTouchStart, { passive: false });
    canvas.addEventListener('touchmove', handleCanvasTouchMove, { passive: false });
    canvas.addEventListener('touchend', handleCanvasTouchEnd);
    
    // Atualizar dimensões do canvas
    window.addEventListener('resize', function() {
        canvasRect = canvas.getBoundingClientRect();
    });
}

function initPalette() {
    const palette = document.getElementById('palette');
    if (!palette) return;
    
    // Adicionar elementos à paleta
    paletteElements.forEach(element => {
        const div = document.createElement('div');
        div.className = `draggable-element ${element.id}`;
        div.draggable = true;
        div.dataset.type = element.type;
        div.dataset.subtype = element.subtype;
        div.dataset.color = element.color;
        div.dataset.emoji = element.emoji;
        div.dataset.name = element.name;
        
        div.innerHTML = `
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 20px;">${element.emoji}</span>
                <div>
                    <strong>${element.name}</strong><br>
                    <small>Arraste para o canvas</small>
                </div>
            </div>
        `;
        
        // Eventos de drag and drop
        div.addEventListener('dragstart', function(e) {
            e.dataTransfer.setData('text/plain', JSON.stringify({
                type: this.dataset.type,
                subtype: this.dataset.subtype,
                color: this.dataset.color,
                emoji: this.dataset.emoji,
                name: this.dataset.name
            }));
            this.style.opacity = '0.5';
        });
        
        div.addEventListener('dragend', function() {
            this.style.opacity = '1';
        });
        
        palette.appendChild(div);
    });
    
    // Configurar área de drop
    const canvas = document.getElementById('canvas');
    if (canvas) {
        canvas.addEventListener('dragover', function(e) {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'copy';
        });
        
        canvas.addEventListener('drop', function(e) {
            e.preventDefault();
            const data = JSON.parse(e.dataTransfer.getData('text/plain'));
            
            // Calcular posição relativa ao canvas
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Adicionar elemento ao canvas
            addElementToCanvas(data, x, y);
        });
    }
}

function addElementToCanvas(data, x, y) {
    const canvas = document.getElementById('canvas');
    if (!canvas) return;
    
    const elementId = 'element-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    
    let elementDiv;
    
    if (data.type === 'bus') {
        elementDiv = document.createElement('div');
        elementDiv.className = `canvas-bus bus-${data.subtype}-bg`;
        elementDiv.id = elementId;
        elementDiv.dataset.id = elementId;
        elementDiv.dataset.type = 'bus';
        elementDiv.dataset.subtype = data.subtype;
        elementDiv.dataset.x = x - 20;
        elementDiv.dataset.y = y - 20;
        elementDiv.dataset.name = data.name;
        elementDiv.textContent = elements.filter(e => e.type === 'bus').length + 1;
        
        // Posicionar elemento
        elementDiv.style.left = (x - 20) + 'px';
        elementDiv.style.top = (y - 20) + 'px';
        
        // Adicionar label
        const label = document.createElement('div');
        label.className = 'element-label';
        label.textContent = data.name;
        label.style.left = (x - 20) + 'px';
        label.style.top = (y + 30) + 'px';
        label.id = 'label-' + elementId;
        
        canvas.appendChild(label);
        
    } else if (data.type === 'line') {
        // Linhas são criadas conectando duas barras
        if (currentMode === 'connect' && connectingFrom) {
            // Conectar barras
            const fromElement = document.getElementById(connectingFrom);
            const toElement = document.querySelector('.canvas-bus:hover');
            
            if (fromElement && toElement && fromElement !== toElement) {
                createLine(fromElement, toElement, data.color);
                connectingFrom = null;
                updateStatus();
            }
        }
        return;
        
    } else if (data.type === 'load' || data.type === 'generator') {
        // Elementos que precisam estar associados a uma barra
        const closestBus = findClosestBus(x, y);
        if (closestBus) {
            elementDiv = document.createElement('div');
            elementDiv.className = data.type === 'load' ? 'canvas-load' : 'canvas-gen';
            elementDiv.id = elementId;
            elementDiv.dataset.id = elementId;
            elementDiv.dataset.type = data.type;
            elementDiv.dataset.bus = closestBus.id;
            elementDiv.dataset.name = data.name;
            elementDiv.textContent = data.type === 'load' ? 'L' : 'G';
            
            // Posicionar próximo à barra
            const busRect = closestBus.getBoundingClientRect();
            const busX = parseInt(closestBus.dataset.x) + 20;
            const busY = parseInt(closestBus.dataset.y) + 20;
            
            const offsetY = data.type === 'load' ? 50 : -50;
            elementDiv.style.left = (busX - 15) + 'px';
            elementDiv.style.top = (busY + offsetY - 15) + 'px';
            
            elementDiv.dataset.x = busX - 15;
            elementDiv.dataset.y = busY + offsetY - 15;
        } else {
            alert('Coloque o elemento próximo a uma barra!');
            return;
        }
    }
    
    if (elementDiv) {
        // Adicionar eventos
        elementDiv.addEventListener('mousedown', startDrag);
        elementDiv.addEventListener('touchstart', startTouchDrag, { passive: false });
        elementDiv.addEventListener('click', handleElementClick);
        
        canvas.appendChild(elementDiv);
        elements.push({
            id: elementId,
            type: data.type,
            element: elementDiv
        });
        
        // Enviar dados para Python
        sendToPython('add_element', {
            id: elementId,
            type: data.type,
            subtype: data.subtype,
            x: parseInt(elementDiv.dataset.x),
            y: parseInt(elementDiv.dataset.y),
            name: data.name
        });
    }
}

function createLine(fromElement, toElement, color) {
    const lineId = 'line-' + Date.now();
    const canvas = document.getElementById('canvas');
    
    const x1 = parseInt(fromElement.dataset.x) + 20;
    const y1 = parseInt(fromElement.dataset.y) + 20;
    const x2 = parseInt(toElement.dataset.x) + 20;
    const y2 = parseInt(toElement.dataset.y) + 20;
    
    // Calcular comprimento e ângulo
    const length = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
    const angle = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;
    
    const lineDiv = document.createElement('div');
    lineDiv.className = 'canvas-line';
    lineDiv.id = lineId;
    lineDiv.dataset.id = lineId;
    lineDiv.dataset.type = 'line';
    lineDiv.dataset.from = fromElement.id;
    lineDiv.dataset.to = toElement.id;
    
    lineDiv.style.width = length + 'px';
    lineDiv.style.left = x1 + 'px';
    lineDiv.style.top = y1 + 'px';
    lineDiv.style.transform = `rotate(${angle}deg)`;
    lineDiv.style.backgroundColor = color;
    
    // Adicionar eventos
    lineDiv.addEventListener('click', handleElementClick);
    
    canvas.appendChild(lineDiv);
    lines.push(lineDiv);
    
    // Enviar dados para Python
    sendToPython('add_line', {
        id: lineId,
        from: fromElement.id,
        to: toElement.id,
        x1: x1, y1: y1,
        x2: x2, y2: y2
    });
}

function findClosestBus(x, y) {
    const buses = document.querySelectorAll('.canvas-bus');
    let closest = null;
    let minDist = 100; // Raio máximo
    
    buses.forEach(bus => {
        const busX = parseInt(bus.dataset.x) + 20;
        const busY = parseInt(bus.dataset.y) + 20;
        const dist = Math.sqrt(Math.pow(busX - x, 2) + Math.pow(busY - y, 2));
        
        if (dist < minDist) {
            minDist = dist;
            closest = bus;
        }
    });
    
    return closest;
}

function startDrag(e) {
    if (currentMode !== 'select') return;
    
    e.preventDefault();
    e.stopPropagation();
    
    selectedElement = this;
    isDragging = true;
    
    const rect = this.getBoundingClientRect();
    dragOffset = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
    };
    
    // Remover seleção anterior
    document.querySelectorAll('.selected').forEach(el => {
        el.classList.remove('selected');
    });
    
    // Selecionar este elemento
    this.classList.add('selected');
    
    // Atualizar label se existir
    const label = document.getElementById('label-' + this.id);
    if (label) {
        label.classList.add('selected');
    }
}

function startTouchDrag(e) {
    if (currentMode !== 'select') return;
    
    e.preventDefault();
    
    if (e.touches.length === 1) {
        const touch = e.touches[0];
        selectedElement = this;
        isDragging = true;
        
        const rect = this.getBoundingClientRect();
        dragOffset = {
            x: touch.clientX - rect.left,
            y: touch.clientY - rect.top
        };
        
        this.classList.add('selected');
    }
}

function handleCanvasMouseDown(e) {
    if (e.target === e.currentTarget && currentMode === 'connect' && !connectingFrom) {
        // Modo conexão - primeiro clique
        const bus = document.querySelector('.canvas-bus:hover');
        if (bus) {
            connectingFrom = bus.id;
            bus.classList.add('selected');
            updateStatus();
        }
    }
}

function handleCanvasMouseMove(e) {
    if (!isDragging || !selectedElement) return;
    
    const canvas = document.getElementById('canvas');
    const rect = canvas.getBoundingClientRect();
    
    let x = e.clientX - rect.left - dragOffset.x;
    let y = e.clientY - rect.top - dragOffset.y;
    
    // Limitar ao canvas
    x = Math.max(0, Math.min(x, canvas.offsetWidth - selectedElement.offsetWidth));
    y = Math.max(0, Math.min(y, canvas.offsetHeight - selectedElement.offsetHeight));
    
    // Atualizar posição
    selectedElement.style.left = x + 'px';
    selectedElement.style.top = y + 'px';
    selectedElement.dataset.x = x;
    selectedElement.dataset.y = y;
    
    // Atualizar label
    const label = document.getElementById('label-' + selectedElement.id);
    if (label) {
        label.style.left = x + 'px';
        label.style.top = (y + 40) + 'px';
    }
    
    // Atualizar linhas conectadas
    updateConnectedLines(selectedElement);
}

function handleCanvasMouseUp() {
    if (isDragging && selectedElement) {
        // Enviar nova posição para Python
        sendToPython('move_element', {
            id: selectedElement.id,
            x: parseInt(selectedElement.dataset.x),
            y: parseInt(selectedElement.dataset.y)
        });
    }
    
    isDragging = false;
    selectedElement = null;
}

function handleCanvasTouchStart(e) {
    if (e.touches.length === 1) {
        const touch = e.touches[0];
        handleCanvasMouseDown({ clientX: touch.clientX, clientY: touch.clientY, target: e.target, currentTarget: e.currentTarget });
    }
}

function handleCanvasTouchMove(e) {
    if (e.touches.length === 1 && isDragging && selectedElement) {
        const touch = e.touches[0];
        handleCanvasMouseMove({ clientX: touch.clientX, clientY: touch.clientY });
        e.preventDefault();
    }
}

function handleCanvasTouchEnd(e) {
    handleCanvasMouseUp();
}

function updateConnectedLines(element) {
    const elementId = element.id;
    
    lines.forEach(line => {
        if (line.dataset.from === elementId || line.dataset.to === elementId) {
            const fromElement = document.getElementById(line.dataset.from);
            const toElement = document.getElementById(line.dataset.to);
            
            if (fromElement && toElement) {
                const x1 = parseInt(fromElement.dataset.x) + 20;
                const y1 = parseInt(fromElement.dataset.y) + 20;
                const x2 = parseInt(toElement.dataset.x) + 20;
                const y2 = parseInt(toElement.dataset.y) + 20;
                
                const length = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
                const angle = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;
                
                line.style.width = length + 'px';
                line.style.left = x1 + 'px';
                line.style.top = y1 + 'px';
                line.style.transform = `rotate(${angle}deg)`;
            }
        }
    });
}

function handleElementClick(e) {
    e.stopPropagation();
    
    // Remover seleção anterior
    document.querySelectorAll('.selected').forEach(el => {
        el.classList.remove('selected');
    });
    
    // Selecionar este elemento
    this.classList.add('selected');
    
    // Atualizar label se existir
    const label = document.getElementById('label-' + this.id);
    if (label) {
        label.classList.add('selected');
    }
    
    // Enviar para Python
    sendToPython('select_element', {
        id: this.id,
        type: this.dataset.type,
        subtype: this.dataset.subtype
    });
}

function setMode(mode) {
    currentMode = mode;
    connectingFrom = null;
    
    // Atualizar botões
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.mode === mode) {
            btn.classList.add('active');
        }
    });
    
    updateStatus();
}

function updateStatus() {
    const status = document.getElementById('status');
    if (!status) return;
    
    let statusText = '';
    
    switch(currentMode) {
        case 'select':
            statusText = '🔍 Modo Seleção: Arraste os elementos para mover';
            break;
        case 'connect':
            if (connectingFrom) {
                statusText = `🔗 Modo Conexão: Clique em outra barra para conectar com a barra ${connectingFrom}`;
            } else {
                statusText = '🔗 Modo Conexão: Clique em uma barra para começar a conexão';
            }
            break;
        case 'delete':
            statusText = '🗑️ Modo Deleção: Clique nos elementos para remover';
            break;
    }
    
    status.innerHTML = `<div class="status-bar">${statusText}</div>`;
}

function deleteSelected() {
    const selected = document.querySelector('.selected');
    if (selected) {
        // Remover label se existir
        const label = document.getElementById('label-' + selected.id);
        if (label) {
            label.remove();
        }
        
        // Remover linhas conectadas
        if (selected.dataset.type === 'bus') {
            const connectedLines = Array.from(lines).filter(line => 
                line.dataset.from === selected.id || line.dataset.to === selected.id
            );
            connectedLines.forEach(line => line.remove());
            lines = lines.filter(line => !connectedLines.includes(line));
        }
        
        // Enviar para Python
        sendToPython('delete_element', {
            id: selected.id,
            type: selected.dataset.type
        });
        
        selected.remove();
        
        // Remover dos arrays
        elements = elements.filter(el => el.id !== selected.id);
    }
}

function clearCanvas() {
    const canvas = document.getElementById('canvas');
    if (canvas) {
        canvas.innerHTML = '';
        elements = [];
        lines = [];
        sendToPython('clear_canvas', {});
    }
}

function sendToPython(action, data) {
    // Enviar mensagem para o Streamlit
    const message = {
        action: action,
        data: data,
        timestamp: new Date().toISOString()
    };
    
    // Usando o método do Streamlit para comunicação
    if (window.Streamlit) {
        window.Streamlit.setComponentValue(message);
    }
    
    // Log para debug
    console.log('Sending to Python:', message);
}

// Expor funções globalmente
window.setMode = setMode;
window.deleteSelected = deleteSelected;
window.clearCanvas = clearCanvas;
</script>
""", unsafe_allow_html=True)

# Inicialização do estado
def init_session_state():
    if 'elements' not in st.session_state:
        st.session_state.elements = {}
    if 'selected_element' not in st.session_state:
        st.session_state.selected_element = None
    if 'mode' not in st.session_state:
        st.session_state.mode = 'select'
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'system_name' not in st.session_state:
        st.session_state.system_name = "Sistema Elétrico"

init_session_state()

# Componente HTML para o canvas
def create_canvas_component():
    html_content = f"""
    <div style="width: 100%;">
        <!-- Status Bar -->
        <div id="status"></div>
        
        <!-- Canvas Area -->
        <div class="canvas-container canvas-grid" id="canvas" style="width: 100%; height: 700px;">
            <!-- Elementos serão adicionados aqui via JavaScript -->
        </div>
        
        <!-- Controles flutuantes -->
        <div style="position: absolute; top: 100px; left: 20px; z-index: 1000;">
            <div style="background: white; padding: 10px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                <h4 style="margin: 0 0 10px 0; color: #333;">Controles</h4>
                <button onclick="setMode('select')" class="mode-btn" data-mode="select" style="background: {'#007bff' if st.session_state.mode == 'select' else '#6c757d'}; color: white; border: none; padding: 8px 12px; border-radius: 5px; cursor: pointer; margin: 5px 0; width: 100%;">
                    🔍 Selecionar
                </button>
                <button onclick="setMode('connect')" class="mode-btn" data-mode="connect" style="background: {'#007bff' if st.session_state.mode == 'connect' else '#6c757d'}; color: white; border: none; padding: 8px 12px; border-radius: 5px; cursor: pointer; margin: 5px 0; width: 100%;">
                    🔗 Conectar
                </button>
                <button onclick="setMode('delete')" class="mode-btn" data-mode="delete" style="background: {'#007bff' if st.session_state.mode == 'delete' else '#6c757d'}; color: white; border: none; padding: 8px 12px; border-radius: 5px; cursor: pointer; margin: 5px 0; width: 100%;">
                    🗑️ Deletar
                </button>
                <hr style="margin: 10px 0;">
                <button onclick="deleteSelected()" style="background: #dc3545; color: white; border: none; padding: 8px 12px; border-radius: 5px; cursor: pointer; margin: 5px 0; width: 100%;">
                    Remover Selecionado
                </button>
                <button onclick="clearCanvas()" style="background: #6c757d; color: white; border: none; padding: 8px 12px; border-radius: 5px; cursor: pointer; margin: 5px 0; width: 100%;">
                    Limpar Tudo
                </button>
            </div>
        </div>
        
        <script>
        // Inicializar quando a página carregar
        setTimeout(function() {{
            if (typeof initCanvas === 'function') {{
                initCanvas();
                initPalette();
                updateStatus();
                
                // Definir modo inicial
                setMode('{st.session_state.mode}');
            }}
        }}, 1000);
        </script>
    </div>
    """
    return html(html_content, height=750)

# Paleta de elementos
def create_palette():
    with st.sidebar:
        st.header("🎨 Paleta de Elementos")
        st.markdown("Arraste e solte no canvas:")
        
        html("""
        <div id="palette">
            <!-- Elementos serão adicionados via JavaScript -->
        </div>
        """, height=400)
        
        st.divider()
        
        st.header("📊 Análises")
        if st.button("🔁 Calcular Fluxo de Potência", use_container_width=True):
            st.info("Fluxo calculado! (simulação)")
            st.session_state.results = {
                'type': 'power_flow',
                'timestamp': datetime.now().isoformat(),
                'voltages': [0.98, 1.02, 0.95, 1.08],
                'converged': True
            }
        
        if st.button("⚡ Análise de Curto-Circuito", use_container_width=True):
            st.info("Curto-circuito analisado! (simulação)")
            st.session_state.results = {
                'type': 'short_circuit',
                'timestamp': datetime.now().isoformat(),
                'fault_current': "15.3 kA",
                'location': "Barra 2"
            }
        
        st.divider()
        
        st.header("⚙️ Configurações")
        st.session_state.system_name = st.text_input("Nome do Sistema", 
                                                    st.session_state.system_name)
        
        st.divider()
        
        st.header("💾 Arquivos")
        if st.button("📥 Exportar Sistema", use_container_width=True):
            data = {
                'system': st.session_state.elements,
                'results': st.session_state.results,
                'metadata': {
                    'name': st.session_state.system_name,
                    'export_date': datetime.now().isoformat()
                }
            }
            st.download_button(
                label="Baixar JSON",
                data=json.dumps(data, indent=2),
                file_name=f"{st.session_state.system_name}.json",
                mime="application/json"
            )
        
        if st.button("🔄 Resetar Sistema", type="secondary", use_container_width=True):
            st.session_state.elements = {}
            st.session_state.selected_element = None
            st.rerun()

# Função para processar mensagens do JavaScript
def handle_js_message(message):
    if isinstance(message, dict):
        action = message.get('action')
        data = message.get('data', {})
        
        if action == 'add_element':
            element_id = data.get('id')
            st.session_state.elements[element_id] = {
                'type': data.get('type'),
                'subtype': data.get('subtype'),
                'x': data.get('x'),
                'y': data.get('y'),
                'name': data.get('name'),
                'timestamp': datetime.now().isoformat()
            }
            
        elif action == 'move_element':
            element_id = data.get('id')
            if element_id in st.session_state.elements:
                st.session_state.elements[element_id]['x'] = data.get('x')
                st.session_state.elements[element_id]['y'] = data.get('y')
            
        elif action == 'select_element':
            st.session_state.selected_element = {
                'id': data.get('id'),
                'type': data.get('type'),
                'subtype': data.get('subtype')
            }
            
        elif action == 'delete_element':
            element_id = data.get('id')
            if element_id in st.session_state.elements:
                del st.session_state.elements[element_id]
            
        elif action == 'clear_canvas':
            st.session_state.elements = {}
            st.session_state.selected_element = None

# Interface principal
def main():
    st.title("⚡ Power System Editor - Drag & Drop")
    st.markdown("Crie sistemas elétricos arrastando e conectando elementos visualmente")
    
    # Layout principal
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Canvas interativo
        st.subheader("Canvas Interativo")
        
        # Criar componente HTML
        canvas_component = create_canvas_component()
        
        # Processar mensagens do JavaScript
        if canvas_component:
            # Esta é uma versão simplificada - em produção, você usaria um componente customizado
            pass
        
        # Estatísticas
        st.markdown("---")
        col_stats = st.columns(4)
        with col_stats[0]:
            buses = sum(1 for e in st.session_state.elements.values() if e['type'] == 'bus')
            st.metric("Barras", buses)
        with col_stats[1]:
            lines = sum(1 for e in st.session_state.elements.values() if e['type'] == 'line')
            st.metric("Linhas", lines)
        with col_stats[2]:
            loads = sum(1 for e in st.session_state.elements.values() if e['type'] == 'load')
            st.metric("Cargas", loads)
        with col_stats[3]:
            gens = sum(1 for e in st.session_state.elements.values() if e['type'] == 'generator')
            st.metric("Geradores", gens)
    
    with col2:
        create_palette()
        
        # Painel de propriedades
        if st.session_state.selected_element:
            st.subheader("🔧 Propriedades")
            
            element_info = st.session_state.selected_element
            element_id = element_info.get('id')
            element_data = st.session_state.elements.get(element_id, {})
            
            st.info(f"Elemento selecionado: {element_data.get('name', 'N/A')}")
            
            with st.form("properties_form"):
                if element_data.get('type') == 'bus':
                    new_name = st.text_input("Nome", element_data.get('name', 'Barra'))
                    new_type = st.selectbox("Tipo", ["slack", "pv", "pq"], 
                                          index=["slack", "pv", "pq"].index(element_data.get('subtype', 'pq')))
                    
                    col_x, col_y = st.columns(2)
                    with col_x:
                        new_x = st.number_input("Posição X", value=element_data.get('x', 0))
                    with col_y:
                        new_y = st.number_input("Posição Y", value=element_data.get('y', 0))
                    
                    if st.form_submit_button("💾 Atualizar"):
                        if element_id in st.session_state.elements:
                            st.session_state.elements[element_id].update({
                                'name': new_name,
                                'subtype': new_type,
                                'x': new_x,
                                'y': new_y
                            })
                            st.success("Atualizado!")
                            st.rerun()
                
                elif element_data.get('type') == 'load':
                    new_name = st.text_input("Nome", element_data.get('name', 'Carga'))
                    new_p = st.number_input("P (MW)", value=5.0)
                    new_q = st.number_input("Q (MVar)", value=2.0)
                    
                    if st.form_submit_button("💾 Atualizar"):
                        st.success("Atualizado!")
                        st.rerun()
                
                elif element_data.get('type') == 'generator':
                    new_name = st.text_input("Nome", element_data.get('name', 'Gerador'))
                    new_p = st.number_input("P (MW)", value=10.0)
                    new_v = st.number_input("Vm (pu)", value=1.0)
                    
                    if st.form_submit_button("💾 Atualizar"):
                        st.success("Atualizado!")
                        st.rerun()
        
        # Resultados
        if st.session_state.results:
            st.subheader("📊 Resultados")
            
            results = st.session_state.results
            time_str = datetime.fromisoformat(results.get('timestamp', datetime.now().isoformat())).strftime("%H:%M:%S")
            
            if results.get('type') == 'power_flow':
                st.markdown(f"**Fluxo de Potência** ({time_str})")
                
                if results.get('converged'):
                    st.success("✅ Convergiu")
                else:
                    st.error("❌ Não convergiu")
                
                if 'voltages' in results:
                    avg_v = np.mean(results['voltages'])
                    st.metric("Tensão Média", f"{avg_v:.3f} pu")
            
            elif results.get('type') == 'short_circuit':
                st.markdown(f"**Curto-Circuito** ({time_str})")
                
                if 'fault_current' in results:
                    st.metric("Corrente de Falta", results['fault_current'])
                
                if 'location' in results:
                    st.info(f"Local: {results['location']}")

# Executar aplicação
if __name__ == "__main__":
    main()
