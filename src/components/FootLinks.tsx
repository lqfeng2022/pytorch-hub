import { HStack, Box, Text } from '@chakra-ui/react'
import { Link } from 'react-router-dom'

interface Props {
  leftLink: string,
  rightLink: string
}
const FootLinks = ({ leftLink, rightLink }: Props) => {
  const hoverStyle = { color: 'tomato', textDecoration: 'underline' }

  return (
    <HStack justifyContent='space-between' pb={5}>
      <Box>
        <Link to={'/introduction'}>
          <Text color='gray'>Previous Chapter</Text>
          <Text as='b' _hover={hoverStyle}>
            {`<- ${leftLink}`}
          </Text>
        </Link>
      </Box>
      <Box>
        <Link to={'/tensors'}>
          <Text textAlign='right' color='gray'>Next Chapter</Text>
          <Text as='b' _hover={hoverStyle}>
            {`${rightLink} ->`}
          </Text>
        </Link>
      </Box>
    </HStack>
  )
}

export default FootLinks