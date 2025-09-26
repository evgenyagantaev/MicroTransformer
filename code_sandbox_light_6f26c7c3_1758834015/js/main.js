/**
 * Главный файл приложения
 * Управляет анимацией и взаимодействием пользователя
 */
class NeuralNetworkDemo {
    constructor() {
        this.network = new NeuralNetwork();
        this.visualization = new Visualization('circleCanvas', 'networkCanvas');
        
        // Состояние анимации
        this.isPlaying = false;
        this.animationId = null;
        this.lastTimestamp = 0;
        this.speed = 1.0; // секунд между кадрами
        
        // Статистика
        this.iterations = 0;
        this.errors = [];
        this.currentError = 0;
        
        // Текущие значения
        this.currentOmega = 0;
        this.currentPrediction = 0;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.updateUI();
        
        // Показать начальное состояние
        this.performStep();
        
        console.log('Демонстрация нейронной сети инициализирована');
        this.network.testNetwork();
    }
    
    setupEventListeners() {
        // Кнопки управления
        document.getElementById('playPauseBtn').addEventListener('click', () => {
            this.togglePlayPause();
        });
        
        document.getElementById('stepBtn').addEventListener('click', () => {
            this.performStep();
        });
        
        document.getElementById('resetBtn').addEventListener('click', () => {
            this.reset();
        });
        
        // Ползунок скорости
        const speedRange = document.getElementById('speedRange');
        speedRange.addEventListener('input', (e) => {
            this.speed = parseFloat(e.target.value);
            document.getElementById('speedValue').textContent = this.speed.toFixed(1) + 'x';
        });
        
        // Обработка изменения размера окна
        window.addEventListener('resize', () => {
            this.visualization.resize();
        });
    }
    
    togglePlayPause() {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }
    
    play() {
        this.isPlaying = true;
        this.updatePlayButton();
        this.lastTimestamp = performance.now();
        this.animate();
    }
    
    pause() {
        this.isPlaying = false;
        this.updatePlayButton();
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
    
    animate(timestamp) {
        if (!this.isPlaying) return;
        
        const deltaTime = timestamp - this.lastTimestamp;
        const interval = (1000 / this.speed); // миллисекунды
        
        if (deltaTime >= interval) {
            this.performStep();
            this.lastTimestamp = timestamp;
        }
        
        this.animationId = requestAnimationFrame((ts) => this.animate(ts));
    }
    
    performStep() {
        // Генерируем случайное значение omega в диапазоне [0, 1]
        this.currentOmega = Math.random();
        
        // Получаем предсказание от сети
        this.currentPrediction = this.network.predict(this.currentOmega);
        
        // Вычисляем ошибку
        this.currentError = Math.abs(this.currentOmega - this.currentPrediction);
        this.errors.push(this.currentError);
        
        // Ограничиваем историю ошибок
        if (this.errors.length > 100) {
            this.errors.shift();
        }
        
        this.iterations++;
        
        // Обновляем визуализацию
        this.updateVisuals();
        this.updateUI();
    }
    
    updateVisuals() {
        // Рисуем векторы на окружности
        this.visualization.drawVectors(this.currentOmega, this.currentPrediction);
        
        // Обновляем визуализацию сети
        const networkInfo = this.network.getLastComputationInfo();
        this.visualization.updateNetwork(networkInfo);
        
        // Подсвечиваем большие ошибки
        this.visualization.pulseError(this.currentError);
    }
    
    updateUI() {
        // Обновляем кнопку воспроизведения
        this.updatePlayButton();
        
        // Обновляем статистику
        document.getElementById('currentError').textContent = this.currentError.toFixed(3);
        document.getElementById('avgError').textContent = this.getAverageError().toFixed(3);
        document.getElementById('iterations').textContent = this.iterations.toString();
        
        // Обновляем текущие значения
        document.getElementById('inputOmega').textContent = this.currentOmega.toFixed(3);
        
        const x1 = Math.cos(2 * Math.PI * this.currentOmega);
        const x2 = Math.sin(2 * Math.PI * this.currentOmega);
        
        document.getElementById('inputCos').textContent = x1.toFixed(3);
        document.getElementById('inputSin').textContent = x2.toFixed(3);
        document.getElementById('outputPrediction').textContent = this.currentPrediction.toFixed(3);
        document.getElementById('outputError').textContent = this.currentError.toFixed(3);
        
        // Добавляем цветовую индикацию ошибки
        const errorElement = document.getElementById('outputError');
        errorElement.classList.remove('error-highlight');
        
        if (this.currentError > 0.1) {
            errorElement.classList.add('error-highlight');
        }
    }
    
    updatePlayButton() {
        const btn = document.getElementById('playPauseBtn');
        const icon = btn.querySelector('i');
        
        if (this.isPlaying) {
            icon.className = 'fas fa-pause';
            btn.innerHTML = '<i class=\"fas fa-pause\"></i> Пауза';
        } else {
            icon.className = 'fas fa-play';
            btn.innerHTML = '<i class=\"fas fa-play\"></i> Старт';
        }
    }
    
    getAverageError() {
        if (this.errors.length === 0) return 0;
        const sum = this.errors.reduce((acc, err) => acc + err, 0);
        return sum / this.errors.length;
    }
    
    reset() {
        this.pause();
        this.iterations = 0;
        this.errors = [];
        this.currentError = 0;
        this.currentOmega = 0;
        this.currentPrediction = 0;
        
        this.updateUI();
        
        // Сбрасываем визуализацию
        this.visualization.drawCircleBackground();
        this.visualization.drawNetworkBackground();
        
        console.log('Демонстрация сброшена');
    }
    
    /**
     * Добавляем интерактивность - клик по canvas для ручного выбора точки
     */
    setupCanvasInteraction() {
        const canvas = document.getElementById('circleCanvas');
        
        canvas.addEventListener('click', (event) => {
            if (this.isPlaying) return; // Не реагируем во время анимации
            
            const rect = canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            
            // Преобразуем координаты клика в omega
            const center = this.visualization.circleCenter;
            const radius = this.visualization.radius;
            
            const dx = x - center.x;
            const dy = center.y - y; // Инвертируем Y
            
            // Проверяем, что клик внутри или рядом с окружностью
            const distance = Math.sqrt(dx*dx + dy*dy);
            if (distance <= radius + 20) {
                // Вычисляем угол
                let angle = Math.atan2(dy, dx);
                if (angle < 0) angle += 2 * Math.PI;
                
                // Преобразуем в omega [0, 1]
                this.currentOmega = angle / (2 * Math.PI);
                
                // Выполняем предсказание
                this.currentPrediction = this.network.predict(this.currentOmega);
                this.currentError = Math.abs(this.currentOmega - this.currentPrediction);
                
                // Обновляем без изменения счетчика итераций
                this.updateVisuals();
                this.updateUI();
                
                console.log(`Ручной выбор: ω = ${this.currentOmega.toFixed(3)}, предсказание = ${this.currentPrediction.toFixed(3)}`);
            }
        });
        
        // Добавляем курсор pointer при наведении на canvas
        canvas.addEventListener('mousemove', (event) => {
            const rect = canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            
            const center = this.visualization.circleCenter;
            const radius = this.visualization.radius;
            
            const dx = x - center.x;
            const dy = y - center.y;
            const distance = Math.sqrt(dx*dx + dy*dy);
            
            if (distance <= radius + 20 && !this.isPlaying) {
                canvas.style.cursor = 'pointer';
            } else {
                canvas.style.cursor = 'default';
            }
        });
    }
    
    /**
     * Добавляем анализ производительности сети
     */
    analyzePerformance() {
        console.log('Анализ производительности сети:');
        console.log('================================');
        
        const testPoints = 100;
        const errors = [];
        
        for (let i = 0; i < testPoints; i++) {
            const omega = i / testPoints;
            const prediction = this.network.predict(omega);
            const error = Math.abs(omega - prediction);
            errors.push(error);
        }
        
        const avgError = errors.reduce((sum, err) => sum + err, 0) / errors.length;
        const maxError = Math.max(...errors);
        const minError = Math.min(...errors);
        
        console.log(`Средняя ошибка: ${avgError.toFixed(4)}`);
        console.log(`Максимальная ошибка: ${maxError.toFixed(4)}`);
        console.log(`Минимальная ошибка: ${minError.toFixed(4)}`);
        
        // Находим самые проблематичные области
        const problematicAreas = [];
        errors.forEach((error, index) => {
            if (error > avgError * 2) {
                problematicAreas.push({
                    omega: index / testPoints,
                    error: error
                });
            }
        });
        
        if (problematicAreas.length > 0) {
            console.log('Проблематичные области:');
            problematicAreas.forEach(area => {
                console.log(`  ω = ${area.omega.toFixed(3)}, ошибка = ${area.error.toFixed(4)}`);
            });
        }
    }
}

// Инициализация приложения при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    window.demo = new NeuralNetworkDemo();
    
    // Добавляем интерактивность
    window.demo.setupCanvasInteraction();
    
    // Выполняем анализ производительности через 2 секунды
    setTimeout(() => {
        window.demo.analyzePerformance();
    }, 2000);
    
    console.log('Приложение готово к использованию!');
    console.log('Используйте кнопки управления или кликните по окружности для интерактивного тестирования.');
});