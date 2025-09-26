/**
 * Класс для реализации нейронной сети
 * Архитектура: 2 входа -> 5 скрытых нейронов -> 1 выход
 */
class NeuralNetwork {
    constructor() {
        // Веса из изображения
        // w1 - матрица весов между входным и скрытым слоем (5x2)
        this.w1 = [
            [-0.027879663, -1.161300981],    // Нейрон 1
            [-3.079965337,	3.261402016],    // Нейрон 2
            [-2.39189672,	4.672663729],    // Нейрон 3
            [4.328507205,	-1.140666928],    // Нейрон 4
            [-0.114288473,	-1.370266297]            // Нейрон 5 
        ];
        
        // b1 - смещения для скрытого слоя (5x1)
        this.b1 = [
            -1.414828364,   // Нейрон 1
            0.700587313,    // Нейрон 2  
            2.24018664,    // Нейрон 3
            -4.771448707,   // Нейрон 4
            0.608089742         // Нейрон 5 
        ];
        
        // w2 - веса между скрытым и выходным слоем (1x5)
        // В изображении показано только 4 веса, добавляем пятый
        this.w2 = [-0.602267397,	1.971134114,	-2.101092697,	1.855199641,	0.982605707];
        
        // b2 - смещение для выходного слоя
        this.b2 = 2.722717113;
        
        // Для отладки и визуализации
        this.lastHiddenValues = [];
        this.lastInputs = [];
        this.lastOutput = 0;
    }
    
    /**
     * Функция активации для скрытого слоя: 2/(1+exp(-h)) - 1
     * Это модифицированная сигмоида с диапазоном [-1, 1]
     */
    hiddenActivation(x) {
        return (2 / (1 + Math.exp(-x)) - 1) * 2;
    }
    
    /**
     * Функция активации для выходного слоя: 1/(1+exp(-z))
     * Стандартная сигмоида с диапазоном [0, 1]
     */
    outputActivation(x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    /**
     * Прямое прохождение через сеть
     * @param {number} x1 - cos(2π*ω)
     * @param {number} x2 - sin(2π*ω)
     * @returns {number} - предсказанное значение ω в диапазоне [0, 1]
     */
    forward(x1, x2) {
        // Сохраняем входы для отладки
        this.lastInputs = [x1, x2];
        
        // Скрытый слой
        const hiddenOutputs = [];
        for (let i = 0; i < 5; i++) {
            // h = w1*x + b1
            const h = this.w1[i][0] * x1 + this.w1[i][1] * x2 + this.b1[i];
            // Применяем функцию активации
            const hiddenOutput = this.hiddenActivation(h);
            hiddenOutputs.push(hiddenOutput);
        }
        
        // Сохраняем значения скрытого слоя для отладки
        this.lastHiddenValues = hiddenOutputs;
        
        // Выходной слой
        // z = w2*h + b2
        let z = this.b2;
        for (let i = 0; i < 5; i++) {
            z += this.w2[i] * hiddenOutputs[i];
        }
        
        // Применяем сигмоиду
        const output = this.outputActivation(z);
        
        // Сохраняем выход для отладки
        this.lastOutput = output;
        
        return output;
    }
    
    /**
     * Предсказание на основе угла ω
     * @param {number} omega - угол в диапазоне [0, 1]
     * @returns {number} - предсказанное значение
     */
    predict(omega) {
        const x1 = Math.cos(2 * Math.PI * omega);
        const x2 = Math.sin(2 * Math.PI * omega);
        return this.forward(x1, x2);
    }
    
    /**
     * Получить информацию о последнем прохождении
     */
    getLastComputationInfo() {
        return {
            inputs: this.lastInputs,
            hiddenValues: this.lastHiddenValues,
            output: this.lastOutput,
            weights: {
                w1: this.w1,
                b1: this.b1,
                w2: this.w2,
                b2: this.b2
            }
        };
    }
    
    /**
     * Вычислить ошибку между предсказанием и истинным значением
     * @param {number} predicted - предсказанное значение
     * @param {number} actual - истинное значение
     * @returns {number} - среднеквадратичная ошибка
     */
    static calculateError(predicted, actual) {
        return Math.abs(predicted - actual);
    }
    
    /**
     * Преобразовать предсказанное значение обратно в координаты на окружности
     * @param {number} omega - предсказанное значение ω
     * @returns {object} - координаты {x, y}
     */
    static omegaToCoordinates(omega) {
        return {
            x: Math.cos(2 * Math.PI * omega),
            y: Math.sin(2 * Math.PI * omega)
        };
    }
    
    /**
     * Тестирование сети на нескольких точках
     */
    testNetwork() {
        console.log('Тестирование нейронной сети:');
        console.log('==============================');
        
        const testCases = [0, 0.25, 0.5, 0.75, 1.0];
        
        testCases.forEach(omega => {
            const prediction = this.predict(omega);
            const error = NeuralNetwork.calculateError(prediction, omega);
            
            console.log(`ω = ${omega.toFixed(3)}`);
            console.log(`  Входы: cos=${Math.cos(2*Math.PI*omega).toFixed(3)}, sin=${Math.sin(2*Math.PI*omega).toFixed(3)}`);
            console.log(`  Предсказание: ${prediction.toFixed(3)}`);
            console.log(`  Ошибка: ${error.toFixed(3)}`);
            console.log('---');
        });
    }
}

// Экспорт для использования в других файлах
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NeuralNetwork;
}