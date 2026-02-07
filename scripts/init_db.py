#!/usr/bin/env python3
"""
初始化数据库表结构
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.postgres import init_db

if __name__ == "__main__":
    print("初始化数据库表...")
    init_db()
    print("完成！")
