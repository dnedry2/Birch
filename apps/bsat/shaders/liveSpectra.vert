#version 410 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoords;

out vec2 TexCoord;
uniform mat4 projection;

void main()
{
    TexCoord = aTexCoords;
    gl_Position = projection * vec4(aPos.xy, 0.0, 1.0); 
}