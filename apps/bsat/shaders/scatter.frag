#version 410 core

uniform sampler2D tex;
uniform vec2      texSize;
uniform vec2      renderSize;

in vec2  TexCoord;
out vec4 FragColor;

void main()
{
    FragColor = texture(tex, TexCoord);
}