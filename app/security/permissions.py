from __future__ import annotations

from enum import Enum

from app.security.auth import AuthenticatedUser
from app.security.roles import UserRole


class Permission(str, Enum):
    ANALYZE_CASE = "analyze_case"
    VIEW_RESULT = "view_result"
    REVIEW_REPORT = "review_report"
    RUN_EVALUATION = "run_evaluation"
    VIEW_AUDIT = "view_audit"
    MANAGE_USERS = "manage_users"


ROLE_PERMISSIONS: dict[UserRole, set[Permission]] = {
    UserRole.VIEWER: {Permission.VIEW_RESULT},
    UserRole.DOCTOR: {
        Permission.ANALYZE_CASE,
        Permission.VIEW_RESULT,
        Permission.REVIEW_REPORT,
    },
    UserRole.RADIOLOGIST: {
        Permission.ANALYZE_CASE,
        Permission.VIEW_RESULT,
        Permission.REVIEW_REPORT,
    },
    UserRole.RESEARCHER: {
        Permission.RUN_EVALUATION,
        Permission.VIEW_RESULT,
    },
    UserRole.ADMIN: {
        Permission.ANALYZE_CASE,
        Permission.VIEW_RESULT,
        Permission.REVIEW_REPORT,
        Permission.RUN_EVALUATION,
        Permission.VIEW_AUDIT,
        Permission.MANAGE_USERS,
    },
}


def has_permission(user: AuthenticatedUser, permission: Permission) -> bool:
    return permission in ROLE_PERMISSIONS.get(user.role, set())
