import { Box, Text, Stack, HStack, Link } from '@chakra-ui/react'

interface ReferLinks {
  id: number,
  title: string,
  value: string,
  link: string,
}

interface Reference {
  id: number,
  name: string,
  values: ReferLinks[],
}

const ReferGrid = ({ section }: { section: Reference }) => {
  return (
    <Box>
      <Text as='b' fontSize='lg' color='gray.600'>
        {section.name}
      </Text>
      <Stack spacing={2} pt={3}>
        {section.values.map((p) => 
          <HStack key={p.id} alignItems='flex-start'>
            <Text as="span" color="gray.400" mr={2} fontSize='xl'>
              •
            </Text>
            <Box>
              <Text as="span">
                <Link
                  href={p.link}
                  textDecoration="underline"
                  textUnderlineOffset='3px'
                  fontStyle='italic'
                  target='_blank'
                  _hover={{ textColor: 'tomato' }}
                >
                  {p.title}
                </Link>
                  { p.value && <Text as='span'>{', '}{p.value}</Text>}
              </Text>
            </Box>
          </HStack>
        )}
      </Stack>
    </Box>
  )
} 

export default ReferGrid