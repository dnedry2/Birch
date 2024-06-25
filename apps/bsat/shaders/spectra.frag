#version 410 core

uniform sampler2D tex;
uniform vec2      texSize;
uniform vec2      renderSize;

in vec2  TexCoord;
out vec4 FragColor;

float luma(vec4 color) {
    return 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
}

void main()
{
    bool shouldScaleY = texSize.y > renderSize.y;
    bool shouldScaleX = texSize.x > renderSize.x;

    vec4 outColor = texture(tex, TexCoord);

    if (shouldScaleY)
    {
        float scale      = 1.0 / (renderSize.y / texSize.y);
        float scaledY = TexCoord.y * scale;

        // Get the width of one texel
        float texelSize = scale / texSize.y;

		vec4 upTexel = texture(tex, vec2(TexCoord.x, TexCoord.y - texelSize));

		float cLuma = luma(outColor);
		float upLuma = luma(upTexel);
		
		if (upLuma > cLuma) {
			outColor = upTexel;
		}

        // Get the max of those texels
        /*
        for (float i = scaledY; i < TexCoord.y; i += texelSize)
        {
            //outColor = max(outColor, texture(tex, vec2(TexCoord.x, i)));
            outColor = vec4(1, outColor.g, outColor.b, outColor.a);
        }
        */
    }

/*
    if (shouldScaleX)
    {
        float scale   = renderSize.x / texSize.x;
        float scaledX = TexCoord.x * scale;

        // Get number of texels to sample
        float texelSize = 1.0 / texSize.x;

        // Get the max of those texels
        for (float i = scaledX; i < TexCoord.y; i += texelSize)
        {
            outColor = max(outColor, texture(tex, vec2(i, TexCoord.y)));
        }
    }
*/
    FragColor = outColor;
}