#!/usr/bin/env python3

import sys
import cv2
import os
from datetime import datetime
from typing import Optional
from breathing_monitor_research import BreathingMonitorResearch
import config

try:
    import snowflake.connector
    from snowflake.connector import DictCursor
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False


class SnowflakeLogger:
    def __init__(self):
        self.conn = None
        self.cursor = None
        self.enabled = config.ENABLE_SNOWFLAKE_LOGGING and SNOWFLAKE_AVAILABLE
        if self.enabled:
            self._connect()
            self._create_table()
    
    def _connect(self):
        try:
            account = config.SNOWFLAKE_ACCOUNT or os.getenv('SNOWFLAKE_ACCOUNT')
            user = config.SNOWFLAKE_USER or os.getenv('SNOWFLAKE_USER')
            password = config.SNOWFLAKE_PASSWORD or os.getenv('SNOWFLAKE_PASSWORD')
            database = config.SNOWFLAKE_DATABASE or os.getenv('SNOWFLAKE_DATABASE')
            schema = config.SNOWFLAKE_SCHEMA or os.getenv('SNOWFLAKE_SCHEMA', 'PUBLIC')
            warehouse = config.SNOWFLAKE_WAREHOUSE or os.getenv('SNOWFLAKE_WAREHOUSE')
            
            if not all([account, user, password, database, warehouse]):
                self.enabled = False
                return
            
            self.conn = snowflake.connector.connect(
                account=account,
                user=user,
                password=password,
                database=database,
                schema=schema,
                warehouse=warehouse
            )
            self.cursor = self.conn.cursor()
        except Exception:
            self.enabled = False
    
    def _create_table(self):
        if not self.enabled or not self.cursor:
            return
        try:
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {config.SNOWFLAKE_TABLE} (
                TIMESTAMP TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                HEART_RATE FLOAT,
                BREATHING_RATE FLOAT
            )
            """
            self.cursor.execute(create_table_sql)
            self.conn.commit()
        except Exception:
            self.enabled = False
    
    def log_vitals(self, monitor: BreathingMonitorResearch):
        if not self.enabled or not self.cursor:
            return False
        if not monitor.is_stabilized:
            return False
        if monitor.heart_rate == 0 and monitor.breathing_rate == 0:
            return False
        try:
            insert_sql = f"""
            INSERT INTO {config.SNOWFLAKE_TABLE} (HEART_RATE, BREATHING_RATE)
            VALUES (%s, %s)
            """
            self.cursor.execute(insert_sql, (monitor.heart_rate, monitor.breathing_rate))
            self.conn.commit()
            return True
        except Exception:
            return False
    
    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()


def main():
    video_path = '/Users/aidenm/Testch/test_videos/002.mp4'
    monitor = BreathingMonitorResearch()
    db_logger = SnowflakeLogger()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_num = 0
    paused = False
    breathing_rates = []
    heart_rates = []
    confidences = []
    hr_confidences = []
    last_db_save_time = 0.0
    db_save_interval = config.SNOWFLAKE_LOG_INTERVAL
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            processed_frame = monitor.process_frame(frame)
            plot_image = monitor.update_plot()
            
            display_frame = cv2.resize(processed_frame, (960, 540))
            display_plot = cv2.resize(plot_image, (960, 540))
            combined = cv2.hconcat([display_frame, display_plot])
            
            progress = frame_num / frame_count
            bar_width = combined.shape[1]
            bar_height = 10
            progress_bar = int(progress * bar_width)
            
            cv2.rectangle(combined, (0, combined.shape[0] - bar_height), 
                         (progress_bar, combined.shape[0]), (0, 255, 0), -1)
            cv2.rectangle(combined, (progress_bar, combined.shape[0] - bar_height),
                         (bar_width, combined.shape[0]), (50, 50, 50), -1)
            
            elapsed = frame_num / fps if fps > 0 else 0
            duration = frame_count / fps if fps > 0 else 0
            info_text = f"Frame: {frame_num}/{frame_count} | Time: {elapsed:.1f}/{duration:.1f}s | Progress: {progress*100:.1f}%"
            cv2.putText(combined, info_text, (10, combined.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Video Analysis', combined)
            
            if monitor.breathing_rate > 0:
                breathing_rates.append(monitor.breathing_rate)
            if monitor.heart_rate > 0:
                heart_rates.append(monitor.heart_rate)
            confidences.append(monitor.confidence)
            hr_confidences.append(monitor.hr_confidence)
            
            if db_logger.enabled and (elapsed - last_db_save_time >= db_save_interval):
                db_logger.log_vitals(monitor)
                last_db_save_time = elapsed
        
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_name = f'analysis_frame_{frame_num}.jpg'
            cv2.imwrite(screenshot_name, combined)
        elif key == ord(' '):
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()
    db_logger.close()

if __name__ == "__main__":
    main()

