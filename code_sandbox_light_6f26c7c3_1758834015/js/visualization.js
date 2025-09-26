/**
 * Класс для визуализации единичной окружности и нейронной сети
 */
class Visualization {
    constructor(circleCanvasId, networkCanvasId) {
        this.circleCanvas = document.getElementById(circleCanvasId);
        this.networkCanvas = document.getElementById(networkCanvasId);
        
        this.circleCtx = this.circleCanvas.getContext('2d');
        this.networkCtx = this.networkCanvas.getContext('2d');
        
        // Настройки для окружности
        this.circleCenter = { x: 200, y: 200 };
        this.radius = 150;
        
        // Цвета
        this.colors = {
            background: '#0f172a',
            circle: '#475569',
            axes: '#64748b',
            grid: '#334155',
            original: '#4CAF50',
            predicted: '#f44336',
            text: '#f8fafc',
            accent: '#6366f1',
            warning: '#f59e0b'
        };
        
        // Настройки для сети
        this.networkLayout = {
            layers: [
                { x: 80, y: 200, neurons: 2, labels: ['cos(2πω)', 'sin(2πω)'] },
                { x: 200, y: 200, neurons: 5, labels: ['H₁', 'H₂', 'H₃', 'H₄', 'H₅'] },
                { x: 320, y: 200, neurons: 1, labels: ['ω'] }
            ],
            neuronRadius: 15,
            neuronSpacing: 30
        };
        
        this.init();
    }
    
    init() {
        this.setupCanvases();
        this.drawCircleBackground();
        this.drawNetworkBackground();
    }
    
    setupCanvases() {
        // Настройка DPI для четкости
        const ratio = window.devicePixelRatio || 1;
        
        [this.circleCanvas, this.networkCanvas].forEach(canvas => {
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * ratio;
            canvas.height = rect.height * ratio;
            canvas.style.width = rect.width + 'px';
            canvas.style.height = rect.height + 'px';
            
            const ctx = canvas.getContext('2d');
            ctx.scale(ratio, ratio);
        });
        
        // Обновляем центр и радиус после масштабирования
        const circleRect = this.circleCanvas.getBoundingClientRect();
        this.circleCenter = { x: circleRect.width / 2, y: circleRect.height / 2 };
        this.radius = Math.min(circleRect.width, circleRect.height) / 3;
    }
    
    /**
     * Рисует фон для окружности с координатными осями
     */
    drawCircleBackground() {
        const ctx = this.circleCtx;
        const { x: cx, y: cy } = this.circleCenter;
        
        // Очистка
        ctx.fillStyle = this.colors.background;
        ctx.fillRect(0, 0, this.circleCanvas.width, this.circleCanvas.height);
        
        // Сетка
        ctx.strokeStyle = this.colors.grid;
        ctx.lineWidth = 0.5;
        ctx.setLineDash([5, 5]);
        
        // Вертикальные линии сетки
        for (let i = 0; i < this.circleCanvas.width; i += 20) {
            ctx.beginPath();
            ctx.moveTo(i, 0);
            ctx.lineTo(i, this.circleCanvas.height);
            ctx.stroke();
        }
        
        // Горизонтальные линии сетки
        for (let i = 0; i < this.circleCanvas.height; i += 20) {
            ctx.beginPath();
            ctx.moveTo(0, i);
            ctx.lineTo(this.circleCanvas.width, i);
            ctx.stroke();
        }
        
        ctx.setLineDash([]);
        
        // Оси координат
        ctx.strokeStyle = this.colors.axes;
        ctx.lineWidth = 2;
        
        // Ось X
        ctx.beginPath();
        ctx.moveTo(cx - this.radius - 20, cy);
        ctx.lineTo(cx + this.radius + 20, cy);
        ctx.stroke();
        
        // Ось Y
        ctx.beginPath();
        ctx.moveTo(cx, cy - this.radius - 20);
        ctx.lineTo(cx, cy + this.radius + 20);
        ctx.stroke();
        
        // Стрелки на осях
        this.drawArrow(ctx, cx + this.radius + 15, cy, 5, 0);
        this.drawArrow(ctx, cx, cy - this.radius - 15, 5, -Math.PI/2);
        
        // Единичная окружность
        ctx.strokeStyle = this.colors.circle;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(cx, cy, this.radius, 0, 2 * Math.PI);
        ctx.stroke();
        
        // Метки на осях
        ctx.fillStyle = this.colors.text;
        ctx.font = '12px Inter';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        // Метки на окружности (0°, 90°, 180°, 270°)
        const angles = [0, Math.PI/2, Math.PI, 3*Math.PI/2];
        const labels = ['0°', '90°', '180°', '270°'];
        
        angles.forEach((angle, i) => {
            const x = cx + this.radius * Math.cos(angle);
            const y = cy - this.radius * Math.sin(angle);
            
            ctx.fillStyle = this.colors.accent;
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, 2 * Math.PI);
            ctx.fill();
            
            // Подпись
            const labelX = cx + (this.radius + 20) * Math.cos(angle);
            const labelY = cy - (this.radius + 20) * Math.sin(angle);
            
            ctx.fillStyle = this.colors.text;
            ctx.fillText(labels[i], labelX, labelY);
        });
        
        // Подписи осей
        ctx.fillText('cos(2πω)', cx + this.radius + 40, cy + 15);
        ctx.fillText('sin(2πω)', cx - 15, cy - this.radius - 30);
    }
    
    /**
     * Рисует стрелку
     */
    drawArrow(ctx, x, y, size, angle) {
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(angle);
        
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(-size, -size/2);
        ctx.lineTo(-size, size/2);
        ctx.closePath();
        ctx.fillStyle = this.colors.axes;
        ctx.fill();
        
        ctx.restore();
    }
    
    /**
     * Рисует фон для топологии сети
     */
    drawNetworkBackground() {
        const ctx = this.networkCtx;
        
        // Очистка
        ctx.fillStyle = this.colors.background;
        ctx.fillRect(0, 0, this.networkCanvas.width, this.networkCanvas.height);
        
        // Подписи слоев
        ctx.fillStyle = this.colors.text;
        ctx.font = '14px Inter';
        ctx.textAlign = 'center';
        
        ctx.fillText('Входной слой', 80, 50);
        ctx.fillText('Скрытый слой', 200, 50);
        ctx.fillText('Выходной слой', 320, 50);
        
        this.drawNetwork();
    }
    
    /**
     * Рисует статичную структуру нейронной сети
     */
    drawNetwork(activations = null) {
        const ctx = this.networkCtx;
        const layout = this.networkLayout;
        
        // Очистка области сети (не трогаем заголовки)
        ctx.fillStyle = this.colors.background;
        ctx.fillRect(0, 70, this.networkCanvas.width, this.networkCanvas.height - 70);
        
        // Рисуем связи между слоями
        this.drawConnections(ctx, layout);
        
        // Рисуем нейроны
        layout.layers.forEach((layer, layerIndex) => {
            this.drawLayer(ctx, layer, layerIndex, activations);
        });
    }
    
    /**
     * Рисует связи между нейронами
     */
    drawConnections(ctx, layout) {
        ctx.strokeStyle = this.colors.grid;
        ctx.lineWidth = 1;
        
        // Связи между входным и скрытым слоем
        const inputLayer = layout.layers[0];
        const hiddenLayer = layout.layers[1];
        
        for (let i = 0; i < inputLayer.neurons; i++) {
            for (let j = 0; j < hiddenLayer.neurons; j++) {
                const startY = this.getNeuronY(inputLayer, i);
                const endY = this.getNeuronY(hiddenLayer, j);
                
                ctx.beginPath();
                ctx.moveTo(inputLayer.x + layout.neuronRadius, startY);
                ctx.lineTo(hiddenLayer.x - layout.neuronRadius, endY);
                ctx.stroke();
            }
        }
        
        // Связи между скрытым и выходным слоем
        const outputLayer = layout.layers[2];
        
        for (let j = 0; j < hiddenLayer.neurons; j++) {
            const startY = this.getNeuronY(hiddenLayer, j);
            const endY = this.getNeuronY(outputLayer, 0);
            
            ctx.beginPath();
            ctx.moveTo(hiddenLayer.x + layout.neuronRadius, startY);
            ctx.lineTo(outputLayer.x - layout.neuronRadius, endY);
            ctx.stroke();
        }
    }
    
    /**
     * Вычисляет Y координату нейрона в слое
     */
    getNeuronY(layer, neuronIndex) {
        const totalHeight = (layer.neurons - 1) * this.networkLayout.neuronSpacing;
        const startY = layer.y - totalHeight / 2;
        return startY + neuronIndex * this.networkLayout.neuronSpacing;
    }
    
    /**
     * Рисует слой нейронов
     */
    drawLayer(ctx, layer, layerIndex, activations) {
        const layout = this.networkLayout;
        
        for (let i = 0; i < layer.neurons; i++) {
            const x = layer.x;
            const y = this.getNeuronY(layer, i);
            
            // Цвет нейрона в зависимости от активации
            let neuronColor = this.colors.circle;
            if (activations && activations[layerIndex] && activations[layerIndex][i] !== undefined) {
                const activation = Math.abs(activations[layerIndex][i]);
                const intensity = Math.min(activation, 1);
                neuronColor = this.interpolateColor('#334155', '#6366f1', intensity);
            }
            
            // Рисуем нейрон
            ctx.fillStyle = neuronColor;
            ctx.strokeStyle = this.colors.text;
            ctx.lineWidth = 2;
            
            ctx.beginPath();
            ctx.arc(x, y, layout.neuronRadius, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
            
            // Подпись
            ctx.fillStyle = this.colors.text;
            ctx.font = '10px Inter';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(layer.labels[i] || '', x, y);
        }
    }
    
    /**
     * Интерполирует между двумя цветами
     */
    interpolateColor(color1, color2, factor) {
        const rgb1 = this.hexToRgb(color1);
        const rgb2 = this.hexToRgb(color2);
        
        const r = Math.round(rgb1.r + factor * (rgb2.r - rgb1.r));
        const g = Math.round(rgb1.g + factor * (rgb2.g - rgb1.g));
        const b = Math.round(rgb1.b + factor * (rgb2.b - rgb1.b));
        
        return `rgb(${r},${g},${b})`;
    }
    
    /**
     * Конвертирует HEX в RGB
     */
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : null;
    }
    
    /**
     * Рисует векторы на окружности
     */
    drawVectors(originalOmega, predictedOmega) {
        const ctx = this.circleCtx;
        const { x: cx, y: cy } = this.circleCenter;
        
        // Перерисовываем фон
        this.drawCircleBackground();
        
        // Исходный вектор (зеленый)
        const origX = cx + this.radius * Math.cos(2 * Math.PI * originalOmega);
        const origY = cy - this.radius * Math.sin(2 * Math.PI * originalOmega);
        
        this.drawVector(ctx, cx, cy, origX, origY, this.colors.original, 'Исходный');
        
        // Предсказанный вектор (красный)
        const predX = cx + this.radius * Math.cos(2 * Math.PI * predictedOmega);
        const predY = cy - this.radius * Math.sin(2 * Math.PI * predictedOmega);
        
        this.drawVector(ctx, cx, cy, predX, predY, this.colors.predicted, 'Предсказание');
        
        // Показываем значения
        ctx.fillStyle = this.colors.text;
        ctx.font = '12px Inter';
        ctx.textAlign = 'left';
        
        ctx.fillText(`ω = ${originalOmega.toFixed(3)}`, 10, 30);
        ctx.fillText(`ω̂ = ${predictedOmega.toFixed(3)}`, 10, 50);
        ctx.fillText(`Ошибка: ${Math.abs(originalOmega - predictedOmega).toFixed(3)}`, 10, 70);
    }
    
    /**
     * Рисует вектор
     */
    drawVector(ctx, startX, startY, endX, endY, color, label) {
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 3;
        
        // Линия вектора
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
        
        // Точка на конце
        ctx.beginPath();
        ctx.arc(endX, endY, 6, 0, 2 * Math.PI);
        ctx.fill();
        
        // Стрелка
        const angle = Math.atan2(endY - startY, endX - startX);
        this.drawArrow(ctx, endX, endY, 8, angle + Math.PI);
        
        // Подпись рядом с точкой
        ctx.fillStyle = color;
        ctx.font = '10px Inter';
        ctx.textAlign = 'center';
        ctx.fillText(label, endX, endY - 15);
    }
    
    /**
     * Обновляет визуализацию сети с новыми активациями
     */
    updateNetwork(networkInfo) {
        if (!networkInfo) return;
        
        const activations = [
            networkInfo.inputs,           // Входной слой
            networkInfo.hiddenValues,     // Скрытый слой
            [networkInfo.output]          // Выходной слой
        ];
        
        this.drawNetwork(activations);
    }
    
    /**
     * Анимация пульса для подсветки ошибки
     */
    pulseError(error) {
        if (error > 0.1) {
            // Добавляем красную рамку при большой ошибке
            const ctx = this.circleCtx;
            ctx.strokeStyle = this.colors.predicted;
            ctx.lineWidth = 4;
            ctx.setLineDash([10, 5]);
            
            ctx.beginPath();
            ctx.rect(5, 5, this.circleCanvas.width - 10, this.circleCanvas.height - 10);
            ctx.stroke();
            
            ctx.setLineDash([]);
        }
    }
    
    /**
     * Изменение размера canvas при изменении размера окна
     */
    resize() {
        this.setupCanvases();
        this.drawCircleBackground();
        this.drawNetworkBackground();
    }
}

// Экспорт для использования в других файлах
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Visualization;
}