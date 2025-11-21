from fastapi import FastAPI, HTTPException # type: ignore
from typing import List, Optional
from enum import IntEnum
from pydantic import BaseModel, Field # type: ignore

api = FastAPI()

class Priority(IntEnum):
    LOW = 3
    MEDIUM = 2
    HIGH = 1
    
class TodoBase(BaseModel):
    todo_name: str = Field(..., min_length=3, max_length=512, description="name of to do")
    todo_description: Optional[str] = Field(..., description="Description of the todo")
    priority: Priority = Field(Priority.LOW, description='Priority of the todo')
    
    
class TodoCreate(TodoBase):
    pass

class Todo(TodoBase):
    todo_id: int = Field(..., description="ID of the todo")

class TodoUpdate(BaseModel):
    todo_name: Optional[str] = Field(None, min_length=3, max_length=512, description="name of to do")
    todo_description: Optional[str] = Field(None, description="Description of the todo")
    priority: Optional[Priority] = Field(None, description='Priority of the todo')
    
    

all_todos = [
    Todo(todo_id=1, todo_name="Buy groceries", todo_description="Milk, Bread, Eggs", priority=Priority.MEDIUM),
    Todo(todo_id=2, todo_name="Read a book", todo_description="Finish reading '1984'", priority=Priority.LOW),
    Todo(todo_id=3, todo_name="Workout", todo_description="Go for a 30-minute run", priority=Priority.HIGH),
    Todo(todo_id=4, todo_name="Call Mom", todo_description="Check in and say hello", priority=Priority.HIGH),
]

@api.get("/")
def index():
    return {"message": "Hello, World!"}

@api.get('/todos/{todo_id}', response_model=Todo)

def get_todo(todo_id: int):
    for todo in all_todos:
        if todo.todo_id == todo_id:
            todo
    raise HTTPException(status_code=404, detail="Todo not found")

@api.get('/todos', response_model=List[Todo])

def get_todos(first_n: int = None):
    if first_n: 
        return {'result': all_todos[:first_n]}
    else:
        return {'result': all_todos}
    
@api.post('/todos', response_model=Todo)
def create_todo(todo: dict):
    new_todo_id = max( todo['todo_id'] for todo in all_todos) + 1
    
    new_todo = {
        'todo_id': new_todo_id,
        'todo_name': todo['todo_name'],
        'todo_description': todo['todo_description']
    }
    
    all_todos.append(new_todo)
    
    raise HTTPException(status_code=404, detail="Todo not found")

@api.put('/todos/{todo_id}', response_model=Todo)
def update_todo(todo_id: int, updated_todo: TodoUpdate):
    for todo in all_todos:
        if todo.todo_id == todo_id:
            if updated_todo.todo_name is not None:
                todo.todo_name = updated_todo.todo_name
            if updated_todo.todo_description is not None:
                todo.todo_description = updated_todo.todo_description
            if updated_todo.priority is not None:
                todo.priority = updated_todo.priority
            return todo
    raise HTTPException(status_code=404, detail="Todo not found")

@api.delete('/todos/{todo_id}', response_model=Todo)
def delete_todo(todo_id: int):
    for index, todo in enumerate(all_todos):
        if todo.todo_id == todo_id:
            deleted_todo = all_todos.pop(index)
            return deleted_todo
    raise HTTPException(status_code=404, detail="Todo not found")