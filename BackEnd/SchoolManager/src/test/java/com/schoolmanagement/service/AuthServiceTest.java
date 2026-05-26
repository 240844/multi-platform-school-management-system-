package com.schoolmanagement.service;

import com.schoolmanagement.dto.auth.AuthDTOs.*;
import com.schoolmanagement.model.User;
import com.schoolmanagement.model.enums.Role;
import com.schoolmanagement.repository.UserRepository;
import com.schoolmanagement.security.JwtService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.crypto.password.PasswordEncoder;

import java.util.Optional;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class AuthServiceTest {

    @Mock private UserRepository userRepository;
    @Mock private PasswordEncoder passwordEncoder;
    @Mock private JwtService jwtService;
    @Mock private AuthenticationManager authenticationManager;

    @InjectMocks private AuthService authService;

    private RegisterRequest registerRequest;
    private User savedUser;

    @BeforeEach
    void setUp() {
        registerRequest = new RegisterRequest(
                "Jan", "Kowalski", "jan@school.com", "password123", Role.TEACHER
        );
        savedUser = User.builder()
                .id(1L).firstName("Jan").lastName("Kowalski")
                .email("jan@school.com").password("encoded")
                .role(Role.TEACHER).build();
    }

    @Test
    void register_shouldReturnTokenWhenEmailIsNew() {
        when(userRepository.existsByEmail("jan@school.com")).thenReturn(false);
        when(passwordEncoder.encode("password123")).thenReturn("encoded");
        when(userRepository.save(any(User.class))).thenReturn(savedUser);
        when(jwtService.generateToken(any(User.class))).thenReturn("jwt-token");

        AuthResponse response = authService.register(registerRequest);

        assertThat(response.token()).isEqualTo("jwt-token");
        assertThat(response.email()).isEqualTo("jan@school.com");
        assertThat(response.role()).isEqualTo("TEACHER");
    }

    @Test
    void register_shouldThrowWhenEmailAlreadyExists() {
        when(userRepository.existsByEmail("jan@school.com")).thenReturn(true);

        assertThatThrownBy(() -> authService.register(registerRequest))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Email already in use");
    }

    @Test
    void login_shouldReturnTokenForValidCredentials() {
        var loginRequest = new LoginRequest("jan@school.com", "password123");
        when(userRepository.findByEmail("jan@school.com")).thenReturn(Optional.of(savedUser));
        when(jwtService.generateToken(savedUser)).thenReturn("jwt-token");

        AuthResponse response = authService.login(loginRequest);

        assertThat(response.token()).isEqualTo("jwt-token");
        verify(authenticationManager).authenticate(any());
    }
}
