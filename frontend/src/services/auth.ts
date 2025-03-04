import axios from 'axios';

const API_URL = 'http://localhost:8000';

export interface UserData {
    email: string;
    password: string;
    full_name?: string;
}

export interface User {
    id: number;
    email: string;
    full_name: string;
    is_active: boolean;
    created_at: string;
}

export interface AuthResponse {
    access_token: string;
    token_type: string;
}

class AuthService {
    async login(email: string, password: string): Promise<AuthResponse> {
        const formData = new FormData();
        formData.append('username', email);
        formData.append('password', password);

        const response = await axios.post<AuthResponse>(`${API_URL}/token`, formData);
        if (response.data.access_token) {
            localStorage.setItem('token', response.data.access_token);
        }
        return response.data;
    }

    async signup(userData: UserData): Promise<User> {
        const response = await axios.post<User>(`${API_URL}/signup`, userData);
        return response.data;
    }

    async getCurrentUser(): Promise<User | null> {
        try {
            const token = localStorage.getItem('token');
            if (!token) return null;

            const response = await axios.get<User>(`${API_URL}/users/me`, {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });
            return response.data;
        } catch (error) {
            this.logout();
            return null;
        }
    }

    logout(): void {
        localStorage.removeItem('token');
    }

    getToken(): string | null {
        return localStorage.getItem('token');
    }

    isAuthenticated(): boolean {
        return !!this.getToken();
    }
}

export const authService = new AuthService();

// Axios interceptor to add token to requests
axios.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem('token');
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
); 