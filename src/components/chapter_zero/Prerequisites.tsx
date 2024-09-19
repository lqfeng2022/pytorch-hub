import { Box, Stack, Text, Image, HStack } from '@chakra-ui/react'
import { prerequisites } from '../../assets/chapter_zero'
import chapterZero from '../../data/chapterZero'

const Prerequisites = () => {
  const preres = chapterZero[6].sections

  return (
    <Box py={5}>
      <Text as='b' fontSize='lg' color='gray.600'>7. Prerequisites</Text>
      <Image py={3} src={prerequisites}/>
      <Stack spacing={4}>
        {preres.map((p) => 
          <Stack key={p.id}>
            <Text as='b'>{p.name}</Text>
            {p.content.map((t) => 
              <div key={t.id}>
                {t.title && 
                  <div>
                    <HStack>
                      <Text>{t.id}</Text>
                      <Text as='b'>{t.title}</Text>
                    </HStack>
                    <Text py={1} pl={4}>{t.value}</Text>
                  </div>
                }
                {!t.title && <Text py={1}>{t.value}</Text>}
              </div>
            )}
          </Stack>
        )}
      </Stack>
    </Box>
  )
}

export default Prerequisites