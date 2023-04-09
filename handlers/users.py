from fastapi import APIRouter

router = APIRouter()

@router.post("/register-account")
async def create_user():
    user = Users.Users(username="vattnaa", password="12312312", email="pchan")
    # db.add(user)
    # db.commit()
    # db.refresh(user)
    return {"user": user}

# Query users
@router.get("/login")
async def get_users(skip: int = 0, limit: int = 10):
    users = db.query(Users.Users).all()
    return users