Python 3.12.8 (tags/v3.12.8:2dc476b, Dec  3 2024, 19:30:04) [MSC v.1942 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import pandas as pd
... import numpy as np
... import random
... import os
... 
... def generate_dataset(num_records=5000):
...     """Генерирует синтетический датасет для SkillShift 4.0"""
...     np.random.seed(42)
...     random.seed(42)
...     
...     employees = [f"EMP_{i:04d}" for i in range(1, 101)]
...     start_date = pd.Timestamp('2025-01-01')
...     
...     data = []
...     for _ in range(num_records):
...         emp_id = random.choice(employees)
...         
...         # Время и тип смены
...         shift_start = start_date + pd.Timedelta(hours=random.randint(0, 8760))
...         shift_type = random.choice(['Day', 'Night'])
...         
...         # Признаки усталости и HR
...         consecutive_nights = np.random.poisson(1) if shift_type == 'Night' else 0
...         consecutive_nights = min(consecutive_nights, 3)
...         
...         hours_since_last = max(12, np.random.normal(24, 8))
...         skill_level = random.randint(1, 10)
...         production_load = random.uniform(50, 100)
...         
...         # Программный IoT: задержка на турникете как маркер усталости
...         base_delay = 30
...         fatigue_factor = (consecutive_nights * 15) + (max(0, 24 - hours_since_last) * 2)
...         turnstile_delay = max(0, base_delay + fatigue_factor + np.random.normal(0, 10))
...         
        # История из MES
        prev_error_rate = random.uniform(0, 0.1)
        
        # Логика целевой переменной (Брак)
        prob_defect = 0.05
        if consecutive_nights >= 2: prob_defect += 0.15
        if hours_since_last < 14: prob_defect += 0.10
        if skill_level < 4: prob_defect += 0.08
        if production_load > 90: prob_defect += 0.05
        if turnstile_delay > 100: prob_defect += 0.07
        
        defect_occurred = 1 if random.random() < min(prob_defect, 1) else 0
        
        data.append({
            'employee_id': emp_id,
            'shift_start_time': shift_start.strftime('%Y-%m-%d %H:%M'),
            'shift_type': shift_type,
            'consecutive_nights': consecutive_nights,
            'hours_since_last_shift': round(hours_since_last, 1),
            'skill_level_score': skill_level,
            'production_load_pct': round(production_load, 1),
            'turnstile_delay_avg_sec': round(turnstile_delay, 1),
            'mes_error_rate_prev': round(prev_error_rate, 3),
            'defect_occurred': defect_occurred
        })
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Генерация датасета SkillShift 4.0...")
    df = generate_dataset()
    
    
    os.makedirs('data/raw', exist_ok=True)
    output_path = 'data/raw/skillshift_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Успешно! Файл сохранен: {output_path}")
    print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} колонок")
    print("\nПример данных:")
